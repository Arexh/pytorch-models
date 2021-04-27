import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def bce_log_loss(output, target):
    return F.binary_cross_entropy(output, target, reduction='mean')

def bce_log_loss_with_weight(output, target, price):
    return F.binary_cross_entropy(output, target, reduction='mean', weight=price)