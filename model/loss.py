import torch.nn.functional as F
import torch.nn as nn
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)

def mse_loss(output, target):
    criterion = nn.MSELoss(reduction='sum')
    mse =criterion(output, target)
    return mse

def masked_mse_loss(output, target, inp, weight=5e-5):
    mask = torch.where((target > 0.) & (inp > 0.), torch.tensor([1.]).cuda(), torch.tensor([0.]).cuda())

    l2 = torch.sum(torch.abs((mask*(output - target))), (2,3,4)) #B, S , C, H , W
    return weight*torch.mean(l2)

def weighted_mse_loss(input, target, weight=1.0):
    return torch.sum(weight * (input - target) ** 2)

