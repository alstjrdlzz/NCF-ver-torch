import torch
import torch.nn as nn


class GMF(nn.Module):
    def __init__(self,
                 M: int,
                 N: int,
                 K: int):
        super().__init__()

        # Layer
        self.user_embedding_layer = nn.Embedding(M, K)
        self.item_embedding_layer = nn.Embedding(N, K)
        self.output_layer = nn.Linear(K, 1)
        
        
    def forward(self, user, item):
        user_embedding = self.user_embedding_layer(user)
        item_embedding = self.item_embedding_layer(item)
        x = user_embedding * item_embedding
        x = self.output_layer(x)
        output = nn.Sigmoid(x)
        return output