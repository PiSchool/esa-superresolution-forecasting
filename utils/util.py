import os#, sys, humanize, psutil, GPUtil
import numpy as np
import torch
from shutil import copyfile, copytree, rmtree
from PIL import Image
from torchvision import transforms, utils
from pathlib import Path
import json
from collections import OrderedDict
from itertools import repeat
import pandas as pd

def replace_nan(img):

    img[img < -1.] = torch.median(img) if torch.median(img) > 0 else 0
    return img


def order_files(rootdir):
    for f in sorted(os.listdir(rootdir)):
        month_id = f.split('.')[0][-4:-2]
        year_id = f.split('.')[0][-8:-4]
        new_dir = os.path.join(rootdir,year_id+'_'+month_id)
        print(new_dir)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        copyfile(os.path.join(rootdir,f), os.path.join(new_dir,f))

def tile(rootdir):
    for d in sorted(os.listdir(rootdir)):
        sub_dir = os.path.join(rootdir,d)
        for f in sorted(os.listdir(sub_dir)):
            tif = os.path.join(rootdir,d,f)
            print(tif)
            os.system("/home/luka/anaconda3/envs/s5p/bin/gdal_retile.py {} -targetDir {} -ps {} {}".format(tif,'/home/luka/programming/PiSchool/tiledseqs',256,256))

def to_seq(rootdir):
    for f in sorted(os.listdir(rootdir)):
        cur = os.path.join(rootdir,f)
        target = '/home/luka/programming/PiSchool/seqs'
        #date = f.split('.')[0][-14:-8]
        apx = f.split('.')[0][-5:]
        new_dir = os.path.join(target, apx)
        print(new_dir)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        copyfile(cur, os.path.join(new_dir, f))

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def chunk_seq(rootdir):

    for d in sorted(os.listdir(rootdir)):
        lst = []
        for f in sorted(os.listdir(os.path.join(rootdir, d))):
            lst.append(f)
        seqs = chunks(lst, 10)
        target = '/home/luka/programming/PiSchool/seq_chunks'
        apx = f.split('.')[0][-5:]

        i = 0
        for s in seqs:
            new_dir = os.path.join(target, apx + '_' + str(i))
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            for f in s:
                cur = os.path.join(rootdir,d,f)
                copyfile(cur, os.path.join(new_dir, f))
            i+=1

def split_data(rootdir):
    for d in sorted(os.listdir(rootdir)):
        apx = d.split('_')[-1]
        print(apx)
        if apx[-1] == '0':
            copytree(os.path.join(rootdir,d), os.path.join('data/seqsnew/test', d))
        else:
            copytree(os.path.join(rootdir,d), os.path.join('data/seqsnew/train', d))

def clean_files(rootdir):
    for d in sorted(os.listdir(rootdir)):
        for f in sorted(os.listdir(os.path.join(rootdir, d))):
            im = Image.open(os.path.join(rootdir,d, f))
            img = np.asarray(im)
            if np.max(img) < -1.:
                rmtree(os.path.join(rootdir, d))
                print("folder {} deleted".format(d))
                break

            # if img.median() < -1:
            #     print("folder {} delted".format(d))
            #     break

def filter_seqs(rootdir):
    transform = transforms.ToTensor()
    i = 0
    # maxv = 0.0120444145798683
    # minv = -3.6974295653635636e-05
    # normalize = lambda x, min, max: (x - min) / (max - min)
    meanlist = []
    stdlist = []
    for d in sorted(os.listdir(rootdir)):

        for f in sorted(os.listdir(os.path.join(rootdir, d))):
            im = Image.open(os.path.join(rootdir, d, f))
            img = np.asarray(im)
            img = transform(img)
            img = replace_nan(img)
            #img = normalize(img, minv, maxv)
            meanlist.append(torch.mean(img))
            stdlist.append(img.std())

    #print("Max: {}, Min: {}".format(max(maxlist), min(minlist)))
    print("mean: {}".format(sum(meanlist)/len(meanlist)))
    print("std: {}".format(sum(stdlist)/len(stdlist)))
        # if max(maxlist)-min(minlist) < 0.01 and max(maxlist) <0.05:
        #     rmtree(os.path.join(rootdir, d))
        #     print("folder {} deleted".format(d), i)
        #     i += 1

def save_tensors(x, savedir, idx, target=False):
    """
    saves image sequence
    args:
    x -- list of frames
    """
    resultdir = os.path.join(savedir,'samples')
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)
    fn = os.path.join(resultdir, 'pred' + str(idx) + '.png') if not target else os.path.join(resultdir, 'target'+str(idx)+'.png')
    np.random.seed(0)
    prefix = 'pred' if not target else 'target'
    for i in range(x.shape[0]):
        f = x[i].cpu().detach()
        fn = os.path.join(resultdir, prefix + str(idx) + '_batch' +str(i) + '.png')
        utils.save_image(f, fn, padding=0, nrow=f.shape[0])
    print('output image saved to ', fn)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)



