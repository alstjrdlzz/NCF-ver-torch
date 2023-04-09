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
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        user, item = torch.split(x, [1, 1], -1)
        user_embedding = self.user_embedding_layer(user)
        item_embedding = self.item_embedding_layer(item)
        x = torch.mul(user_embedding, item_embedding)
        x = self.output_layer(x)
        output = self.sigmoid(x)
        return output
