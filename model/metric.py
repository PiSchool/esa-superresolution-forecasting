import torch
import torch.nn as nn
from math import log10
import pytorch_ssim

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def PSNR(output, target):
    with torch.no_grad():
        criterion = nn.MSELoss()
        mse = criterion(output, target)
        psnr = 10 * log10(1 / mse.item())
        return psnr

def SSIM(output, target):
    ssim = pytorch_ssim.SSIM(window_size=11)
    total_ssim = 0.
    n_frames = target.shape[1]
    for f in range(n_frames):
        total_ssim += ssim(output[:,f], target[:,f])
    return total_ssim/n_frames