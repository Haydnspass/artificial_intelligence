import argparse
import dill
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from autoencoder import Autoencoder

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


def train(epoch, lossfunction):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = lossfunction(output, data)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
                
    return loss.data[0]


def test(lossfunction):
    model.eval()
    test_loss = 0
    
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
            
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += lossfunction(output, data)#.data[0] # sum up batch loss
        sample =output[0, :]
        truth = data[0,:]
        # pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        # correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    #test_loss /= len(test_loader.dataset)
    print('\nTest set: loss: {}\n'.format(test_loss.data[0]))
    return test_loss.data[0], sample, truth


def plot_evolution_of_sample(data, evolution):
    pass
    
def plot_evolution_test_training(training, testing):
    pass
                
if __name__ == '__main__':

    args = Args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)


    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Autoencoder()
    if args.cuda:
        model.cuda()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    
    best_loss = float('inf')
    
    train_evo = []
    test_evo = []
    sample_evo = []
    truth_evo = []
    
    for epoch in range(1, args.epochs + 1):
        train_evo.append(train(epoch, criterion))
        loss, sample, truth = test(criterion)
        test_evo.append(loss)
        sample_evo.append(sample)
        truth_evo.append(truth)
        
        if loss < best_loss:
            print('New best loss achieved: {:.2f}\n'.format(loss))
            best_loss = loss
            torch.save(model.state_dict(), 'best_model_2.pth')

    filename = 'globalsave2.pkl'
    dill.dump_session(filename)
