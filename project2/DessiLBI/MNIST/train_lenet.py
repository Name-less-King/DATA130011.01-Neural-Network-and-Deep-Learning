import os
#from slbi_toolbox import SLBI_ToolBox
from slbi_toolbox_adam import SLBI_ToolBox
import torch
from utils import *
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import random
from lenet5 import LeNet5


batchsize = 128
max_epoch = 20
lr = 3e-4
kappa = 1
mu = 20
weight_decay = 0
interval = 10
betas = (0.9,0.999)
eps = 1e-8
torch.backends.cudnn.benchmark = True

# set random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(0)

# use GPU or CPU
device = torch.device('cuda:0')
model = LeNet5().to(device)

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,shuffle=True, num_workers=0)
testset = torchvision.datasets.MNIST(root='./data', train=False,download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize,shuffle=False, num_workers=0)

name_list = []
layer_list = []
for name, p in model.named_parameters():
    name_list.append(name)
    print(name)
    if len(p.data.size()) == 4 or len(p.data.size()) == 2:
        layer_list.append(name)
#optimizer = SLBI_ToolBox(model.parameters(), lr=lr, kappa=kappa, mu=mu, weight_decay=weight_decay)
optimizer = SLBI_ToolBox(model.parameters(), lr=lr, kappa=kappa, mu=mu, betas=betas, eps=eps, weight_decay=weight_decay)
optimizer.assign_name(name_list)
optimizer.initialize_slbi(layer_list)

all_num = max_epoch * len(trainloader)
print('num of all step:', all_num)
print('num of step per epoch:', len(trainloader))
for ep in range(max_epoch):
    model.train()
    descent_lr(lr, ep, optimizer, interval)
    loss_val = 0
    correct = num = 0
    for iter, pack in enumerate(trainloader):
        data, target = pack[0].to(device), pack[1].to(device)
        logits = model(data)
        loss = F.nll_loss(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, pred = logits.max(1)
        loss_val += loss.item()
        correct += pred.eq(target).sum().item()
        num += data.shape[0]
        if (iter + 1) % 50 == 0:
            print('*******************************')
            print('epoch : ', ep + 1)
            print('iteration : ', iter + 1)
            print('loss : ', loss_val/100)
            print('Correct : ', correct)
            print('Num : ', num)
            print('Train ACC : ', correct/num)
            correct = num = 0
            loss_val = 0
    optimizer.update_prune_order(ep)
    print('Test Model')
    evaluate_batch(model, testloader, device)
save_model_and_optimizer(model, optimizer, 'lenet.pth')

















