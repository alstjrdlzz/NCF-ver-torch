import torch


@torch.no_grad()
def accuracy(output, target):
    pred = torch.argmax(output, dim=1)
    assert pred.shape[0] == len(target)
    correct = 0
    correct += torch.sum(pred == target).item()
    return correct / len(target)