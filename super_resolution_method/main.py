import os
import argparse

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


import gdal
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from math import log10

from model import Net
from dataset import Dataset
from trainer import Trainer

import pdb
from PIL import Image
from utils import *

parser = argparse.ArgumentParser(description='Subpixel cnn')
parser.add_argument('--scale_factor', type=float, required=True,
                    help="super resolution upscale factor")
parser.add_argument('--num_channels', type=int, required=True,
                    help="number of channels")
parser.add_argument('--batchSize', type=int, default=12,
                    help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=12,
                    help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=10,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Learning Rate. Default=0.0001')
parser.add_argument('--cuda', action='store_true',
                    help='use cuda?')
parser.add_argument('--threads', type=int, default=4,
                    help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed to use. Default=123')
parser.add_argument('--HR', type=str, required=True,
                    help='path to HR data')
parser.add_argument('--LR', type=str, required=True,
                    help='path to LR data')
parser.add_argument('--saveFolder', type=str, required=True,
                    help='saving folder')
opt = parser.parse_args()


if opt.cuda and not torch.cuda.is_available():
  raise Exception("No GPU found, please run --without --cuda")

torch.manual_seed(opt.seed)


device = torch.device("cuda" if opt.cuda else "cpu")

#net
model = Net(upscale_factor = opt.scale_factor)

#weight
model.weight_init()

#opti
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

HR_folder = opt.HR
LR_folder = opt.LR
saveFolder = opt.saveFolder

transform = None
toTensor = transforms.ToTensor()
trainBsize = opt.batchSize
testBsize = opt.testBatchSize
EPOCHS = opt.nEpochs


print('Loading datasets')

Trainer = Trainer(model,EPOCHS, optimizer, HR_folder, LR_folder, saveFolder,
                  transform, toTensor, trainBsize, testBsize, device)
                  
train_loader, test_loader = Trainer.load_dataset()
Trainer.train(train_loader)
Trainer.test(test_loader)
