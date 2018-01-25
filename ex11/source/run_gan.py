import argparse
import dill
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

from model import Discriminator, Generator

class Args:
    def __init__(self):
        self.batch_size = 64
        self.epochs = 10
        self.lr = 0.001
        self.momentum = 0.5
        self.seed = np.random.randint(32000)
        self.log_interval = 100


def train(data_iterator, loss, genr, discr):
    label_real_batch = label_real * torch.ones(args.batch_size)
    label_fake_batch = label_fake * torch.ones(args.batch_size)

    for img in data_iterator:
        # resize

        # zero grads
        genr.zero_grad()
        discr.zero_grad()

        z = Variable(torch.nn.randn(args.batch_size, 100))

        # discriminator forward
        dis_out = discr(img).squeeze()
        loss_real = loss(dis_out, label_real_batch)

        dis_out = dis(gen(z)).squeeze()
        loss_fake = loss(dis_out, label_fake_batch)

        # Backward.
        dis_loss = real_loss + fake_loss
        dis_loss.backward()
        dis_opt.step()

        # Initialize noise.
        z = torch.randn(args.batch_size, 100)

        # Forward.
        dis_out = dis(gen(z)).squeeze()
        gen_loss = loss(dis_out, label_real_batch)

        # Backward.
        gen_loss.backward()
        gen_opt.step()


if __name__ == '__main__':
    args = Args()
    torch.manual_seed(args.seed)

    # load the dataset
    data_mnist = datasets.MNIST('data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    data_loader = iter(DataLoader(data_mnist,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  drop_last=True))

    img_dim = 8 * 8

    # Instanciate generator and discriminator
    gen = Generator()
    gen.weights_init
    dis = Discriminator()
    dis.weights_init

    opt_gen = torch.optim.Adam(gen.parameters(), lr=args.lr)
    opt_dis = torch.optim.Adam(dis.parameters(), lr=args.lr)

    label_real = 1
    label_fake = 0

    loss = torch.nn.MSELoss()

    z_const = torch.randn(64, 100)

    for epoch in range(args.epochs):
        train(data_loader, loss, gen, dis)
