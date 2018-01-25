import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.autograd import Variable


class Generator(nn.Module):

    def __init__(self, input_dim=100, output_dim=784):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.generator = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(self.output_dim, 784),
            nn.Sigmoid()
        )

    def weights_init(self, mu=0, sig=0.075):
        for m in self._modules:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mu, sig**2)
                m.bias.data.normal_(mu, sig**2)

    def forward(self, z):
        z = z.view(-1)

        return self.generator(z)


class Discriminator(nn.Module):

    def __init__(self, input_dim=784, output_dim=1):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.discriminator = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim),
            nn.Sigmoid()
        )

    def weights_init(self, mu=0, sig=0.075):
        for m in self._modules:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mu, sig**2)
                m.bias.data.normal_(mu, sig**2)

    def forward(self, x):
        x = x.view(-1)

        return self.discriminator(x)
