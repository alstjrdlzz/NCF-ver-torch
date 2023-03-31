from typing import List

import torch
import torch.nn as nn


class NeuralCFLayer(nn.Module):
    """
    i: (int) index of layer
    K: (int) embedding size
    layer: (List) number of nodes list, ex) [1024, 256, 64]
    """
    def __init__(self,
                 i: int,
                 K: int,
                 layer: List):
        super().__init__()
    
        # Layer
        if i == 0:
            self.linear = nn.Linear(2*K, layer[i])
        else:
            self.linear = nn.Linear(layer[i-1], layer[i])

    def forward(self, x):
        x = self.linear(x)
        output = nn.ReLU(x)
        return output

class MLP(nn.Module):
    """
    M: (int) the number of users
    N: (int) the number of items
    K: (int) embedding size
    layer: (List) the number of nodes list, ex) [1024, 256, 64]
    """
    def __init__(self,
                 M: int,
                 N: int,
                 K: int,
                 layer: List):
        super().__init__()

        n_layers = len(layer)

        # Layer
        self.user_embedding_layer = nn.Embedding(M, K)
        self.item_embedding_layer = nn.Embedding(N, K)
        self.neural_cf_layers = nn.ModuleList([NeuralCFLayer(i, K, layer) for i in range(n_layers)])
        self.output_layer = nn.Linear(self.layer[-1], 1)

    def forward(self, user, item):
        user_embbedding = self.user_embedding_layer(user)
        item_embbedding = self.item_embedding_layer(item)
        x = torch.concat([user_embbedding, item_embbedding], dim=1)
        for layer in self.neural_cf_layers:
            x = layer(x)
        x = self.output_layer(x)
        output = nn.Sigmoid(x)
        return output
    


