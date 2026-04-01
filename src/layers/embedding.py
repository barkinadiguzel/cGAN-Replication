import torch
import torch.nn as nn

class LabelEmbedding(nn.Module):
    def __init__(self, num_classes, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
    
    def forward(self, y):
        return self.embedding(y)
