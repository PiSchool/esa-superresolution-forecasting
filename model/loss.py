import torch.nn.functional as F
import torch.nn as nn
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)

def mse_loss(output, target):
    criterion = nn.MSELoss(reduction='sum')
    mse =criterion(output, target)
    return mse


def masked_mse_loss(output, target, weight=0.00005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask = torch.where((target > 0.), torch.tensor([1.]).to(device), torch.tensor([0.]).to(device))
    criterion = nn.MSELoss(reduction='none')
    mse = criterion(output, target)
    mse_masked = mse*mask

    #l2 = torch.sum(torch.abs((mask*(output - target))), (2,3,4))
    #return torch.mean(mse_masked)

    return mse_masked.sum() / torch.nonzero(mse_masked.data).size(0)

def weighted_mse_loss(input, target, weight=1.0):
    return torch.sum(weight * (input - target) ** 2)

