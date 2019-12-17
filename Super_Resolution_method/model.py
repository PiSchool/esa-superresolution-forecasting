import torch
import torch.nn as nn
import torch.nn.init as init
import pdb

class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.upscale_factor = int(upscale_factor)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, kernel_size = 5, padding = 2)
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
        self.conv2_3 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(32, int(self.upscale_factor **2), kernel_size = 3, padding = 1)
        self.pixel_shuffle = nn.PixelShuffle(self.upscale_factor)
        self.conv5 = nn.Conv2d(1,1, kernel_size = 1, padding = 0)
        self.weight_init()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2_1(out))
        out = self.relu(self.conv2_2(out))
        out = self.relu(self.conv2_3(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = self.pixel_shuffle(out)
        out = self.conv5(out)
        return out

    def weight_init(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2_1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2_2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2_3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv5.weight)
