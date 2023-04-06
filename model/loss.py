import torch.nn as nn


def bce_loss(output, target):
    return nn.BCELoss(output, target)