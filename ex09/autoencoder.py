import torch
import torch.nn as nn


def to_img(x):
    """ Maps a 2D tensor from range [-1, 1] to 4D tensor with range [0, 1].
    Useful for plotting of reconstructions.

    :param x: 2D Tensor that is supposed to be converted
    :return: Converted 4D Tensor with b, c, w, h, where w = h = 28
    """
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


def add_white_noise(x, factor=0.5, stddev=1):
    """ Adds white noise to an input tensor.
    To make sure that data is in intended range [min, max], use torch.clamp(x, min, max) after applying this function.

    :param x: ND Tensor that is altered
    :param factor: A factor that controls the strength of the additive noise
    :param stddev: The stddev of the normal distribution used for generating the noise
    :return: ND Tensor, x with white noise
    """
    # add white noise to tensor
    noise = torch.normal(means=torch.zeros_like(x), std=(stddev*torch.ones_like(x))) #x.clone().normal_(0, stddev)
    return x + noise * factor


class Autoencoder(nn.Module):

    def __init__(self, input_shape=(28, 28), noise_add=False):
        super(Autoencoder, self).__init__()
        
        self.input_dim = input_shape
        self.encoder_firstDim = input_shape[0] * input_shape[1] #torch.prod(self.input_dim)
        self.encoder_lastDim = 8
        self.decoder_firstDim = self.encoder_lastDim
        self.noise_add = noise_add
        
        self.encoder = nn.Sequential(
            nn.Linear(self.encoder_firstDim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.encoder_lastDim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.decoder_firstDim, 8),
            nn.ReLU(),
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, x):
        
        x = x.view(x.size(0),-1)
        
        if self.noise_add:
            x = add_white_noise(x)
        
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x
        
    def encode(self, x):
        x = x.view(x.size(0),-1)
        x = self.encoder(x)
        return x
