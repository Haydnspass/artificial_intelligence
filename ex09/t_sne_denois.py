import argparse
import dill
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from autoencoder import Autoencoder

def load_data():

        data = datasets.FashionMNIST('data', train=True, download=True,
            transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ]))

        loader = torch.utils.data.DataLoader(dataset=data, batch_size=500, sampler=torch.utils.data.sampler.SequentialSampler(data))

        return loader


if __name__ == '__main__':

    model = Autoencoder(noise_add=True)
    model.load_state_dict(torch.load('best_model_denoising.pth'))
    
    loader = load_data()

    data, label = next(iter(loader))
    label = label.numpy()

    data = Variable(data)
    data_after_encoder = (model.encode(data)).data.numpy()

    # Use t-SNE to reduce dimensionality.
    data_2d = TSNE(n_components=2).fit_transform(data_after_encoder)

    plt.figure()
    plt.scatter(data_2d[:,0],data_2d[:,1], c=label)

    plt.title('t-SNE Visualization of Fashion MNIST')
    plt.savefig('t_sne')
