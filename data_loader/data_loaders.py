from torchvision import datasets, transforms
from base import BaseDataLoader
from data.cams import S5Dataset
from data.mmnist import MovingMNIST
from utils import replace_nan
import torch


class S5DataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, im_size, shuffle=True, validation_split=0.0, num_workers=1, drop_last=True, training=True, n_past=5, n_fut=5):
        MIN_VALUE = 0
        MAX_VALUE = 1.
        trsfm = transforms.Compose([
            transforms.CenterCrop(im_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: replace_nan(x)),
            transforms.Lambda(lambda x: x / 1e-4)
        ])
        self.data_dir = data_dir
        self.dataset = S5Dataset(self.data_dir, transform=trsfm, seqlen=n_past+n_fut, n_past=n_past, train=training)
        print(len(self.dataset))
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, drop_last)

class MMnistDataLoader(BaseDataLoader):
   def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, drop_last=True):
        trsfm = transforms.Compose([
            transforms.ToTensor()
        ])
        self.data_dir = data_dir
        self.dataset = MovingMNIST(train=training, data_root=self.data_dir, seq_len=10,  transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, drop_last)