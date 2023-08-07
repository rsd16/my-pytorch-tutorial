import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import time



class ResBlock(nn.Module):
    def __init__(self, in_size:int, hidden_size:int, out_size:int, pad:int):
        super().__init__()
        self.conv1 = conv(in_size, hidden_size, pad)
        self.conv2 = conv(hidden_size, out_size, pad)
        self.batchnorm1 = nn.BatchNorm2d(hidden_size)
        self.batchnorm2 = nn.BatchNorm2d(out_size)
    
    def convblock(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        return x
    
    def forward(self, x): return x + self.convblock(x) # skip connection