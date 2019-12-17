# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 10:01:24 2019

@author: Maximilien_Houel
"""
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

import pdb
from PIL import Image
from utils import *
 
class Trainer() :
    def __init__(self, model,EPOCHS,optimizer,HR_folder,LR_folder,saveFolder,transform,
                 toTensor, trainBsize, testBsize,
                 device) :
        self.model = model
        self.EPOCHS = EPOCHS
        self.optimizer = optimizer
        self.HR_folder = HR_folder
        self.LR_folder = LR_folder
        self.saveFolder = saveFolder
        self.transform = transform
        self.toTensor = toTensor
        self.trainBsize = trainBsize
        self.testBsize = testBsize
        self.device = device
        self.writer = SummaryWriter(self.saveFolder)
    
    def load_dataset(self) :

        ds = Dataset(self.HR_folder,self.LR_folder, 
                     self.transform, self.toTensor)
        val_pct = 0.7
        train_size = int(len(ds) * val_pct)
        test_size = len(ds) - train_size
    
        print("Train size: {}, Test size: {}".format(train_size, test_size))
        trainset, testset = torch.utils.data.random_split(ds,
                                                          [train_size,
                                                           test_size])
        train = torch.utils.data.DataLoader(trainset,
                                            batch_size = self.trainBsize,
                                            shuffle = True)
        test = torch.utils.data.DataLoader(testset,
                                              batch_size = self.testBsize,
                                              shuffle = False)
        return train, test

    def train(self,train_loader):
          for epoch in range(1 , self.EPOCHS+1) :
            avg_psnr = 0
            epoch_loss = 0
            for iter, batch in enumerate(train_loader) :
              X, y = batch[0].to(self.device), batch[1].to(self.device)
              out = self.model(X)
              
              self.optimizer.zero_grad()
              loss = mse_loss(out,y)
              psnr = PSNR(out, loss)
              epoch_loss += loss.item()
              loss.backward()
              self.optimizer.step()
              avg_psnr += psnr
              print("Epoch: [{}] [{}/{}] loss: {}".format((epoch + 1),
                                                        (iter + 1),
                                                        len(train_loader),
                                                        loss.item()))

            self.writer.add_scalar('Loss/train', epoch_loss / len(train_loader), epoch)

            print("===> Epoch {} Complete: Avg. Loss: {}".format(epoch,
                                                              epoch_loss / len(train_loader)))
            print("===> Epoch {} Complete: Avg. PSNR: {}".format(epoch,
                                                              avg_psnr / len(train_loader)))
    
            if epoch == self.EPOCHS - 1:
                model_out_path = os.path.join(self.saveFolder,
                          "subpixel_epoch_{}.pth".format(epoch))
                torch.save(self.model,
                     model_out_path)
                print("Checkpoint saved to {}".format(model_out_path))
          self.writer.close()
          
    def test(self,test_loader):
          avg_psnr = 0
          with torch.no_grad():
            for iter, batch in enumerate(test_loader) :
              X, y = batch[0].to(self.device), batch[1].to(self.device)

              prediction = self.model(X)

              pred = prediction.detach.numpy()
              targ = y.detach.numpy()
              inp = X.detach.numpy()

              for p in range(pred.shape[0]):
                  pred_img = Image.fromarray(pred[p])
                  targ_img = Image.fromarray(targ[p])
                  in_img = Image.fromarray(inp[p])
                  dst_pred = '{}/prediction_{}_{}.png'.format(self.saveFolder, iter, p)
                  dst_target = '{}/target_{}_{}.png'.format(self.saveFolder, iter, p)
                  dst_input = '{}/input_{}_{}.png'.format(self.saveFolder, iter, p)
                  pred_img.imsave(dst_pred)
                  targ_img.imsave(dst_target)
                  in_img.imsave(dst_input)



              loss = mse_loss(prediction,y)
              psnr = PSNR(prediction, loss)
              avg_psnr += psnr
              print('Loss [{}/{}] = {}'.format(iter, len(test_loader, loss.item())))
              print('PSNR [{}/{}] = {}'.format(iter, len(test_loader), psnr))

              self.writer.add_scalar('PSNR/test', avg_psnr / len(test_loader), iter)

          print("Avg. PSNR : {} dB".format(avg_psnr / len(test_loader)))

