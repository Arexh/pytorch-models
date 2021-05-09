import torch.nn.functional as F
import numpy as np
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)

def bce_log_loss(output, target):
    [m] = output.shape
    if torch.isnan(output).any():
        print("Found nan")
        return output
    return (-1 * torch.sum((target * torch.log(output) + (1 - target) * torch.log(1 - output)))) / m

def bce_log_loss_with_weight(output, target, price):
    [m] = output.shape
    if torch.isnan(output).any():
        print(price)
        print("Found nan")
        return output
    return (-1 * torch.sum((target * torch.log(output) + (1 - target) * torch.log(1 - output) * price))) / m