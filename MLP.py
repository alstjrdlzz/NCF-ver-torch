from typing import List

import torch
import torch.nn as nn


class MLPLayers(nn.Module):
    """
    keras는 dense(layer[0]) + dense(layer[1]) + dense(layer[2]) + ... 이런식으로 구현돰
    torch는 Linear(embe*2, layer[0]) + Linear(layer[0], layer[1]) + ... + Linear(layer[n-2], layer[n-1])
    """
    def __init__(self,
                 layer: List):
        
        super().__init__()

        self.layer = layer
        for i in range(len(layer)):
            self.mlp_layers.append(nn.Linear(layer[i], layer[]))
        

    def forward(self, x):

        return


