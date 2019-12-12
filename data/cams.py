import os
from PIL import Image
import numpy as np
import torch
from utils import *

class S5Dataset(object):
  def __init__(self,root_dir, transform=None, seqlen=30, n_past=5, size=256, train=False):
    self.root_dir = root_dir
    self.transform = transform
    self.dirs = []
    self.train = train
    if self.train:
      for d1 in os.listdir(os.path.join(self.root_dir,'train')):
        self.dirs.append(os.path.join(self.root_dir,'train', d1))
    else:
      for d1 in os.listdir(os.path.join(self.root_dir,'test')):
        self.dirs.append(os.path.join(self.root_dir,'test', d1))
    self.seqlen = seqlen
    self.d = 0
    self.size = size
    self.seed_is_set = False

    self.n_past = n_past

  def set_seed(self, seed):
    if not self.seed_is_set:
      self.seed_is_set = True
      np.random.seed(seed)

  def get_seq(self):
    d = self.dirs[self.d]
    if self.d == len(self.dirs) - 1:
      self.d = 0
    else:
      self.d += 1
    seq = []
    i = 0
    dirlist = sorted(os.listdir(d))[:self.seqlen]
    for fname in dirlist:
      im = Image.open(os.path.join(d,fname))
      im = self.transform(im)
      seq.append(im)
      i += 1
      if i>=self.seqlen:
        break
    full_seq = torch.stack(seq)
    (input, target) = torch.split(full_seq, (self.n_past, self.seqlen-self.n_past), dim=0)
    return input, target

  def __getitem__(self, index):
    self.set_seed(index)
    try:
      return self.get_seq()
    except Exception as e:
      print(e)

  def __len__(self):
      return len(self.dirs)



