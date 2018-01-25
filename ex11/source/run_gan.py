import argparse
import dill
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from torch.autograd import Variable

from model import Discriminator, Generator

class Args:
    def __init__(self):
        self.batch_size = 64
        self.epochs = 10
        self.lr = 0.001
        self.momentum = 0.5
        self.seed = np.random.randint(32000)
        self.log_interval = 1


def train(data_iterator, loss, genr, discr, opt_gen, opt_dis):
    label_real_batch = Variable(label_real * torch.ones(args.batch_size))
    label_fake_batch = Variable(label_fake * torch.ones(args.batch_size))
    for img, _ in data_iterator:

        img = Variable(img.view(args.batch_size, -1))

        # zero grads
        genr.zero_grad()
        discr.zero_grad()

        z = Variable(torch.randn(args.batch_size, 100))

        # discriminator forward
        dis_out = discr(img).squeeze()
        loss_real = loss(dis_out, label_real_batch)

        dis_out = dis(gen(z)).squeeze()
        loss_fake = loss(dis_out, label_fake_batch)

        # Backward.
        loss_dis = loss_real + loss_fake
        loss_dis.backward()
        opt_dis.step()

        # Initialize noise.
        z = Variable(torch.randn(args.batch_size, 100))

        # Forward.
        dis_out = dis(gen(z)).squeeze()
        loss_gen = loss(dis_out, label_real_batch)

        # Backward.
        loss_gen.backward()
        opt_gen.step()


if __name__ == '__main__':
    args = Args()
    torch.manual_seed(args.seed)

    # load the dataset
    data_mnist = datasets.MNIST('data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    data_loader = DataLoader(data_mnist,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  drop_last=True)

    img_dim = 8 * 8

    # Instanciate generator and discriminator
    gen = Generator()
    gen.weights_init()
    dis = Discriminator()
    dis.weights_init()

    opt_gen = torch.optim.Adam(gen.parameters(), lr=args.lr)
    opt_dis = torch.optim.Adam(dis.parameters(), lr=args.lr)

    label_real = 1
    label_fake = 0

    loss = torch.nn.MSELoss()

    z_const = Variable(torch.randn(args.batch_size, 100))

    for epoch in range(args.epochs):
        train(data_loader, loss, gen, dis, opt_gen, opt_dis)

        if epoch % args.log_interval == 0: # show Progress.
            gen_out = gen(z_const)
            gen_out = gen_out.view(-1, 1, 28, 28)
            utils.save_image(gen_out.data, 'generated_at_epoch_{}.png'.format(epoch))

    # Save results.
    torch.save(gen.state_dict(), 'gen_state_dict.pth')
    torch.save(dis.state_dict(), 'dis_state_dict.pth')
