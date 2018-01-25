import argparse
import dill
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from model import Discriminator, Generator

class Args:
    def __init__(self):
        self.batch_size = 128
        self.test_batch_size = 1000
        self.epochs = 10
        self.lr = 0.001
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = np.random.randint(32000)
        self.log_interval = 100


def train(epoch, args):
    pass


if __name__ == '__main__':
    args = Args()
    torch.manual_seed(args.seed)

    
