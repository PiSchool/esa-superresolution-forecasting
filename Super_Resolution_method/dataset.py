# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:35:01 2019

@author: Maximilien_Houel
"""

import os
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import gdal
from tqdm import tqdm
import numpy as np
from PIL import Image

from utils import *
import pdb

class Dataset(data.Dataset):
  def __init__(self, HR_folder, LR_folder, transform = None,toTensor = None):
    self.HR_folder = HR_folder
    self.LR_folder = LR_folder
    self.transform = transform
    self.toTensor = toTensor
    self.min_cams = 4.0738445e-07
    self.max_cams = 2e-05
    self.min_s5p = 0.0
    self.max_s5p = 2e-05

    self.HR_files= [os.path.join(self.HR_folder,x) for x in sorted(os.listdir(self.HR_folder))if x.endswith(".tif")]
    self.LR_files = [os.path.join(self.LR_folder,x) for x in sorted(os.listdir(self.LR_folder))if x.endswith(".tif")]

  def __getitem__(self,indx):

     HR_img = Image.open(self.HR_files[indx])
     LR_img = Image.open(self.LR_files[indx])
     HR_array = np.array(HR_img)
     LR_array = np.array(LR_img)
     LR_bicubic = bicubic_upsampling(LR_img,3)
     HR_bicubic = bicubic_downsampling(HR_img,2)
     LR_norm = normalize(np.array(LR_bicubic), self.min_cams, self.max_cams)
     #LR_norm = normalize(LR_array, self.min_cams, self.max_cams)
     #HR_norm = normalize(np.array(HR_bicubic), self.min_s5p, self.max_s5p)
     HR_norm = normalize(HR_array, self.min_s5p, self.max_s5p)
     HR_tensor = self.toTensor(HR_norm)
     LR_tensor = self.toTensor(LR_norm)
     return LR_tensor, HR_tensor

  def __len__(self):
    return len(self.HR_files)
