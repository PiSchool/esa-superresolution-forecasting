import gdal
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import shutil


import torch
import torchvision
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


from math import log10

def order_files(rootdir, typedata, v):
  for f in sorted(os.listdir(rootdir)):
      if f.endswith('.tif'):
        if typedata == 'S5P' :
          date = f.split('_')[-1:][0][:-4]
          new_dir = rootdir + '/{}_{}_{}'.format(typedata,v,date)
          if not os.path.isdir(new_dir):
            os.mkdir(new_dir)
        else :
          date = f.split('_')[-2:][0]
          new_dir = rootdir + '/{}_{}_{}'.format(typedata,v,date)
          if not os.path.isdir(new_dir):
            os.mkdir(new_dir)
        shutil.copy(rootdir + '/{}'.format(f),
                    new_dir + '/{}'.format(f))
        os.remove(rootdir +'/{}'.format(f))


def resize(dst, src, width, height):
    gdal.Warp(destNameOrDestDS = dst,
              srcDSOrSrcDSTab  = src,
              width = width,
              height = height) 
    
def save_tiff(filename, nparray, ref_proj, ref_geom, nodata_value =-9999):
    proj_ref = gdal.Open(ref_proj, gdal.GA_ReadOnly)
    geom_ref = gdal.Open(ref_geom, gdal.GA_ReadOnly)
    
    [cols, rows] = nparray.shape
    outdata = gdal.GetDriverByName("GTiff").Create(str(filename),rows, cols, 1, gdal.GDT_Float32)
    
    outdata.SetGeoTransform(geom_ref.GetGeoTransform())
    outdata.SetProjection(proj_ref.GetProjectionRef())
    outdata.GetRasterBand(1).SetNoDataValue(nodata_value)
    outdata.GetRasterBand(1).WriteArray(nparray)
    outdata.FlushCache()


def tiling(src, dst, size_w, size_h) :
  tile = 0
  im = Image.open(src)
  w,h = im.size
  for i in range(0,w,size_w) :
    for j in range(0,h,size_h) :
         tile +=1
         destName = dst + '{}.tif'.format(tile)
         if not os.path.isfile(destName) :
              gdal.Translate(destName = destName,
                            srcDS = src,
                            format = 'GTiff',
                            srcWin = [i,j,size_w,size_h])
         
         
def remove_nans(array) :
    clean_data = np.where(array < -1, 0, array)
    return clean_data

def check_hour(array):
  hour = array.split('_')[-2:][0]
  return hour

def bicubic_upsampling(img, scale_factor):
    height = int(img.size[1] * scale_factor)
    width = int(img.size[0] * scale_factor)
    new_size = (width, height)
    bicubic_img = img.resize(new_size, 3)
    return bicubic_img
    
def bicubic_downsampling(img, scale_factor):
    height = int(img.size[1] / scale_factor)
    width = int(img.size[0] / scale_factor)
    new_size = (width, height)
    bicubic_img = img.resize(new_size, 3)
    return bicubic_img



def normalize(array, minimum, maximum) :
  normalized = np.where(array != 0, ((array-minimum)/(maximum-minimum)), 0)
  return normalized


def PSNR(X,loss):
  psnr = 10 * log10(1/loss.item())
  return psnr

def SSIM(X, y):
  ssim_loss = pytorch_ssim.SSIM(window_size=11)
  ssim = ssim_loss(X,y)
  return ssim

def count_zeros(array) :
  num_zeros = (array == 0).sum()
  percent = (100 * num_zeros) / (array.shape[0] * array.shape[1])
  return percent

def count_nans(array) :
  num_zeros = (array < 0).sum()
  percent = (100 * num_zeros) / (array.shape[0] * array.shape[1])
  return percent

def mse_loss(output, target):
    criterion = nn.MSELoss()
    return criterion(output, target)

def masked_mse_loss(output, target):
    mask = torch.where((target > 0.),
                      torch.tensor([1.]),
                      torch.tensor([0.]))
     
    l2 = torch.sum(torch.abs((mask*(output - target)))**2 , (1,2,3))
    return torch.mean(l2)


def loader(path_input, min, max) :
  toTensor = transforms.ToTensor()
  img = Image.open(path_input)
  array = np.array(img)
  norm = normalize(array, min, max)
  data = toTensor(norm)
  data = data.view(-1,1,data.shape[1], data.shape[2])
  return data

def save_tensor(dst,src,nrow):
  utils.save_image(src,dst,nrow = nrow, padding = 0)
  
