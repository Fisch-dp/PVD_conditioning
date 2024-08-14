import torch
from pprint import pprint
from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
from metrics.evaluation_metrics import compute_all_metrics, EMD_CD

import torch.nn as nn
import torch.utils.data

import argparse
from torch.distributions import Normal

from utils.file_utils import *
from utils.visualize import *
from model.pvcnn_generation import PVCNN2Base
from modules import SharedMLP, PVConv
from modules.pvconv import CrossAttention, Attention
from tqdm import tqdm

from datasets.shapenet_data_pc import ShapeNet15kPointClouds

from ldm.util import instantiate_from_config
from ldm.modules.ema import LitEma
from omegaconf import OmegaConf

from torchvision.utils import save_image
import copy
import cv2
from torch.autograd import Variable
from train_generation import GaussianDiffusion, Model
'''
models
'''
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                + (mean1 - mean2)**2 * torch.exp(-logvar2))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    # Assumes data is integers [0, 1]
    assert x.shape == means.shape == log_scales.shape
    px0 = Normal(torch.zeros_like(means), torch.ones_like(log_scales))

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 0.5)
    cdf_plus = px0.cdf(plus_in)
    min_in = inv_stdv * (centered_x - .5)
    cdf_min = px0.cdf(min_in)
    log_cdf_plus = torch.log(torch.max(cdf_plus, torch.ones_like(cdf_plus)*1e-12))
    log_one_minus_cdf_min = torch.log(torch.max(1. - cdf_min,  torch.ones_like(cdf_min)*1e-12))
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
    x < 0.001, log_cdf_plus,
    torch.where(x > 0.999, log_one_minus_cdf_min,
             torch.log(torch.max(cdf_delta, torch.ones_like(cdf_delta)*1e-12))))
    assert log_probs.shape == x.shape
    return log_probs



class PVCNN2(PVCNN2Base):
    sa_blocks = [
        # conv_configs, sa_configs
        # (out_channels, num_blocks, voxel_resolution), (num_centers, radius, num_neighbors, out_channels)
        # ()
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (256, 256, 512))),
    ]
    fp_blocks = [
        # fp_configs, conv_configs
        # (,), (out_channels, num_blocks, voxel_resolution)
        ((256, 256), (256, 3, 8)),
        ((256, 256), (256, 3, 8)),
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(self, num_classes, embed_dim, use_att,use_ca,dropout, extra_feature_channels=3, width_multiplier=1,
                 voxel_resolution_multiplier=1):
        super().__init__(
            num_classes=num_classes, embed_dim=embed_dim, use_att=use_att,use_ca=use_ca,
            dropout=dropout, extra_feature_channels=extra_feature_channels,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )



def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)
    elif schedule_type == 'warm0.1':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.2':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.5':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    else:
        raise NotImplementedError(schedule_type)
    return betas

def get_constrain_function(ground_truth, mask, eps, num_steps=1):
    '''

    :param target_shape_constraint: target voxels
    :return: constrained x
    '''
    # eps_all = list(reversed(np.linspace(0,np.float_power(eps, 1/2), 500)**2))
    eps_all = list(reversed(np.linspace(0, np.sqrt(eps), 1000)**2 ))
    def constrain_fn(x, t):
        eps_ =  eps_all[t] if (t<1000) else 0
        for _ in range(num_steps):
            x  = x - eps_ * ((x - ground_truth) * mask)


        return x
    return constrain_fn


#############################################################################

def get_dataset(dataroot, npoints,category,use_mask=False):
    tr_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
        categories=[category], split='train',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        random_subsample=True, use_mask = use_mask)
    te_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
        categories=[category], split='val',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
        use_mask=use_mask
    )
    return tr_dataset, te_dataset



def evaluate_gen(opt, ref_pcs, logger):

    if ref_pcs is None:
        _, test_dataset = get_dataset(opt.dataroot, opt.npoints, opt.category, use_mask=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size,
                                                      shuffle=False, num_workers=int(opt.workers), drop_last=False)
        ref = []
        for data in tqdm(test_dataloader, total=len(test_dataloader), desc='Generating Samples'):
            x = data['test_points'].transpose(1,2).float().cuda()
            m, s = data['mean'].float().cuda(), data['std'].float().cuda()
            

            ref.append(x*s + m)

        ref_pcs = torch.cat(ref, dim=0).contiguous()

    logger.info("Loading sample path: %s"
      % (opt.eval_path))
    sample_pcs = torch.load(opt.eval_path).contiguous()

    logger.info("Generation sample size:%s reference size: %s"
          % (sample_pcs.size(), ref_pcs.size()))


    # Compute metrics
    results = compute_all_metrics(sample_pcs, ref_pcs, opt.batch_size)
    results = {k: (v.cpu().detach().item()
                   if not isinstance(v, float) else v) for k, v in results.items()}
    pprint(results)
    logger.info(results)

    jsd = JSD(sample_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
    pprint('JSD: {}'.format(jsd))
    logger.info('JSD: {}'.format(jsd))



def generate(model, opt):

    _, test_dataset = get_dataset(opt.dataroot, opt.npoints, opt.category)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size,
                                                  shuffle=False, num_workers=int(opt.workers), drop_last=False)

    with torch.no_grad():

        samples = []
        ref = []

        for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Generating Samples'):
            x = data['test_points'].transpose(1,2).cuda()
            m, s = data['mean'].float().cuda(), data['std'].float().cuda()
            img = data['img'].float().cuda()
            img = img.view(-1, *img.shape[-3:])

            gen = model.gen_samples(x.shape,
                                       'cuda', img=img, clip_denoised=False)

            gen = gen.transpose(1,2).contiguous()
            x = x.transpose(1,2).contiguous()



            gen = gen * s + m
            x = x * s + m
            for i in range(len(img)): save_image(img[i], os.path.join(str(Path(opt.eval_path).parent), f'gt{i}.png'))
            samples.append(gen)
            ref.append(x)
            visualize_pointcloud_batch(os.path.join(str(Path(opt.eval_path).parent), 'gt_x.png'), x[:1], None,
                                       None, None)
            visualize_pointcloud_batch(os.path.join(str(Path(opt.eval_path).parent), 'x.png'), gen[:1], None,
                                       None, None)
                
                                
        samples = torch.cat(samples, dim=0)
        ref = torch.cat(ref, dim=0)
        torch.save(samples, opt.eval_path)



    return ref
    

def main(opt):

    if opt.category == 'airplane':
        opt.beta_start = 1e-5
        opt.beta_end = 0.008
        opt.schedule_type = 'warm0.1'

    exp_id = os.path.splitext(os.path.basename(__file__))[0]
    dir_id = os.path.dirname(__file__)
    output_dir = get_output_dir(dir_id, exp_id)
    copy_source(__file__, output_dir)
    logger = setup_logging(output_dir)

    outf_syn, = setup_output_subdirs(output_dir, 'syn')

    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = Model(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)
    if opt.cuda:
        model.cuda()

    def _transform_(m):
        return nn.parallel.DataParallel(m)
        
    model = model.cuda()
    model.multi_gpu_wrapper(_transform_)

    model.eval()

    with torch.no_grad():

        logger.info("Resume Path:%s" % opt.model)

        resumed_param = torch.load(opt.model)
        model.build(resumed_param['model_state'])


        ref = None
        if opt.generate:
            opt.eval_path = os.path.join(outf_syn, 'samples.pth')
            Path(opt.eval_path).parent.mkdir(parents=True, exist_ok=True)
            ref=generate(model, opt)
            
        if opt.eval_gen:
            # Evaluate generation
            evaluate_gen(opt, ref, logger)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='ShapeNetCore.v2.PC15k/')
    parser.add_argument('--category', default='airplane')

    parser.add_argument('--batch_size', type=int, default=25, help='input batch size')
    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--niter', type=int, default=3000, help='number of epochs to train for')

    parser.add_argument('--generate',default=True)
    parser.add_argument('--eval_gen', default=True)

    parser.add_argument('--nc', default=3)
    parser.add_argument('--npoints', default=2048)
    '''model'''
    parser.add_argument('--beta_start', default=0.0001)
    parser.add_argument('--beta_end', default=0.02)
    parser.add_argument('--schedule_type', default='linear')
    parser.add_argument('--time_num', default=1000)

    #params
    parser.add_argument('--attention', default=True)
    parser.add_argument('--dropout', default=0.1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--loss_type', default='mse')
    parser.add_argument('--model_mean_type', default='eps')
    parser.add_argument('--model_var_type', default='fixedsmall')


    parser.add_argument('--model', default='/cluster/51/go25dap/PVD/output/train_generation/2024-08-03-12-28-28/epoch_3999.pth', help="path to model (to continue training)")
    #'/cluster/51/go25dap/PVD/output/train_generation/2024-07-21-20-24-37/epoch_74.pth'
    parser.add_argument('--autoencoder', default='/cluster/51/go25dap/FinetuneVAE-SD/epoch=19-step=31779.ckpt', help="path to autoencoder model")
    parser.add_argument('--autoencoder_config', default="/cluster/51/go25dap/FinetuneVAE-SD/models/first_stage_models/kl-f16/config.yaml")

    '''eval'''

    parser.add_argument('--eval_path',
                        default='')

    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')

    parser.add_argument('--gpu', type=int, default=0, metavar='S', help='gpu id (default: 0)')

    opt = parser.parse_args()

    if torch.cuda.is_available():
        opt.cuda = True
    else:
        opt.cuda = False

    return opt
if __name__ == '__main__':
    opt = parse_args()
    set_seed(opt)

    main(opt)
