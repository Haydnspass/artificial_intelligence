import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable

from startercode import reparameterize

class Args:
    def __init__(self):
        self.batch_size = 128
        self.test_batch_size = 1000
        self.epochs = 50
        self.lr = 0.001
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = np.random.randint(32000)
        self.log_interval = 100
        
class VariationalAutoEncoder(nn.Module):

    def __init__(self, input_shape=(28, 28), noise_add=False):
        super(VariationalAutoEncoder, self).__init__()
        
        self.input_dim = input_shape[0] * input_shape[1]
        self.encoder_firstDim = self.input_dim #torch.prod(self.input_dim)
        self.encoder_lastDim = 64
        self.decoder_firstDim = self.encoder_lastDim
        self.noise_add = noise_add
        
        self.encoder = nn.Sequential(
            nn.Linear(self.encoder_firstDim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        self.mu = nn.Linear(256, 64)
        self.sigma = nn.Linear(256, 64)
        
        self.decoder = nn.Sequential(
            nn.Linear(self.decoder_firstDim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.encoder_firstDim),
            nn.Tanh()
        )
    
    def forward(self, x):
        
        x = x.view(x.size(0),-1)
        z = self.encoder(x)
        
        mu  = self.mu(z)
        sig = self.sigma(z)
        
        z = reparameterize(self, mu=mu, logvar=sig)
        x_hat = self.decoder(z)
        
        return x_hat, mu, sig
        
    def encode(self, x):
        x = x.view(x.size(0),-1)
        x = self.encoder(x)
        return x
