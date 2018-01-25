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

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, self.output_dim)

    def weights_init(self, mu=0, sig=0.075):
        for m in self._modules:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mu, sig**2)
                m.bias.data.normal_(mu, sig**2)

    def forward(self, z):
        z = z.view(z.size(0),-1)

        z = self.fc1(z)
        z = F.relu(z)
        z = self.fc2(z)
        z = F.sigmoid(z)

        return z


class Discriminator(nn.Module):

    def __init__(self, input_dim=784, output_dim=1):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim


        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, self.output_dim)


    def weights_init(self, mu=0, sig=0.075):
        for m in self._modules:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mu, sig**2)
                m.bias.data.normal_(mu, sig**2)

    def forward(self, x):
        x = x.view(x.size(0),-1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.sigmoid(x)

        return x
