import socket
import numpy as np
import torch
from torchvision import datasets, transforms


# source: https://github.com/edenton/svg
class MovingMNIST(object):
    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, train, data_root, transform=None, seq_len=20, num_digits=2, image_size=64,
                 deterministic=True, subsample=False):
        path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.image_size = image_size
        self.step_length = 0.1
        self.digit_size = image_size // 2
        self.deterministic = deterministic
        self.seed_is_set = False  # multi threaded loading
        self.channels = 1
        self.tf = transform
        self.subsample = subsample
        self.data = datasets.MNIST(
            path,
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(self.digit_size),
                 transforms.ToTensor()]))

        self.N = len(self.data)
        print(self.N)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,
                      image_size,
                      image_size,
                      self.channels),
                     dtype=np.float32)
        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, _ = self.data[idx]

            sx = np.random.randint(image_size - digit_size)
            sy = np.random.randint(image_size - digit_size)
            dx = np.random.randint(-4, 5)
            dy = np.random.randint(-4, 5)
            a = self.digit_size
            for t in range(self.seq_len):
                if sy < 0:
                    sy = 0
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(1, 5)
                        dx = np.random.randint(-4, 5)
                elif sy >= image_size - a:
                    sy = image_size - a - 1
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(-4, 0)
                        dx = np.random.randint(-4, 5)

                if sx < 0:
                    sx = 0
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(1, 5)
                        dy = np.random.randint(-4, 5)
                elif sx >= image_size - a:
                    sx = image_size - a - 1
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(-4, 0)
                        dy = np.random.randint(-4, 5)

                x[t, sy:sy + a, sx:sx + a, 0] += digit.numpy().squeeze()
                sy += dy
                sx += dx

        x[x > 1] = 1.
        x = torch.from_numpy(x).permute(0,3,1,2)
        (input, target) = torch.split(x, (5,5), dim=0)
        if self.subsample:
            x = x[:, ::2, ::2]
        return (input, target)