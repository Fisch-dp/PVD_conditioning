import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class TextEmbedding(nn.Module):
    def __init__(self):#, embed_dim):
        super(TextEmbedding, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        #self.fc = nn.Linear(768, embed_dim)  # BERT base output is 768

    def forward(self, text, device):
        encoded_input = self.tokenizer(
            text, 
            return_tensors='pt', 
            padding=True, 
            truncation=True).to(device)
        
        output = self.bert(**encoded_input)
        #embeddings = output.last_hidden_state[:, 0, :]  # Use the [CLS] token
        return output.last_hidden_state#self.fc(embeddings)     