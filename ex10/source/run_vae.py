import argparse
import dill
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from startercode import loss_function
from vae import VariationalAutoEncoder

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

        
def train(epoch, lossfunction, args):
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
            
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output, mu, sig = model(data)
        
        loss = lossfunction(output, data, mu, sig, args.batch_size, 28, 1)
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
                
    return loss.data[0]

    
if __name__ == '__main__':

    args = Args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)


    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data',
                       train=True,
                       download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
                                  
    model = VariationalAutoEncoder()
    if args.cuda:
        model.cuda()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = loss_function
    
    best_loss = float('inf')
    
    train_evo = []
    
    for epoch in range(1, args.epochs + 1):
        train_evo.append(train(epoch, criterion, args))
        
        torch.save(model.state_dict, 'state_dict/network_epoch_{}.pth'.format(epoch))
