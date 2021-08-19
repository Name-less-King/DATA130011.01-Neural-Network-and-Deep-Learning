import os
from slbi_toolbox import SLBI_ToolBox
# from slbi_toolbox_adam import SLBI_ToolBox
import torch
from utils import *
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import random
from vgg import VGG_A_BatchNorm as VGG


batchsize = 128
max_epoch = 160
kappa = 1
mu = 500
weight_decay = 5e-4


torch.backends.cudnn.benchmark = True
# I no longer use interval for I choose scheduler.step
# interval = 30




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
model = VGG().to(device)

# transformation of input data
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  
    transforms.RandomHorizontalFlip(),  
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize,shuffle=False, num_workers=0)

name_list = []
layer_list = []
for name, p in model.named_parameters():
    name_list.append(name)
    print(name)
    if len(p.data.size()) == 4 or len(p.data.size()) == 2:
        layer_list.append(name)

criterion = nn.CrossEntropyLoss()

# for SGD
lr = 1e-1
optimizer = SLBI_ToolBox(model.parameters(), lr=lr, kappa=kappa, mu=mu,weight_decay=weight_decay)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [80,120], gamma=0.1)

'''
# for Adam
lr = 3e-4
betas = (0.9,0.999)
eps = 1e-8
optimizer = SLBI_ToolBox(model.parameters(), lr=lr, kappa=kappa, mu=mu, betas=betas, eps=eps, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=40,eta_min = 1e-4)
'''

optimizer.assign_name(name_list)
optimizer.initialize_slbi(layer_list)

all_num = max_epoch * len(trainloader)
print('num of all step:', all_num)
print('num of step per epoch:', len(trainloader))
for ep in range(max_epoch):
    model.train()
    # descent_lr(lr, ep, optimizer, interval)
    loss_val = 0
    correct = num = 0
    for iter, pack in enumerate(trainloader):
        data, target = pack[0].to(device), pack[1].to(device)
        logits = model(data)
        loss = criterion(logits, target)
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

    if scheduler is not None:
        scheduler.step()

    print('Test Model')
    evaluate_batch(model, testloader, device)

save_model_and_optimizer(model, optimizer, 'vgg.pth')

















