from __future__ import print_function
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import copy
import numpy as np

#from models import *
from lenet5 import LeNet5
# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

# args = parser.parse_args()
def get_lenet():
    __input_dir__ = "./"
    # __output_dir__ = "./small_model/"
    # if not os.path.isdir(__output_dir__):
    #     os.mkdir(__output_dir__)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.ToTensor()])

    transform_test = transforms.Compose([
        transforms.ToTensor()])

    # testset = torchvision.datasets.MNIST(root=r'./data', train=False, download=True,
    #                                      transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=8)

    # Model
    print('==> Building model..')

    net = LeNet5()
    #load_pth = torch.load('lenet_sgd.pth')
    #net.load_state_dict(load_pth['model'])
    #load_pth = torch.load('lenet_adam.pth')
    #net.load_state_dict(load_pth['model'])
    #load_pth = torch.load('lenet_adam_prune.pth')
    #net.load_state_dict(load_pth)
    load_pth = torch.load('lenet.pth')
    net.load_state_dict(load_pth['model'])
    net = net.to(device)


    device = torch.device('cpu')
    net = net.to(device)
    return net



