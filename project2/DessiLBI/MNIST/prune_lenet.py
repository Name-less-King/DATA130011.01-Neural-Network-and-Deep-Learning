import os
from slbi_toolbox import SLBI_ToolBox
import torch
from utils import *
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
from lenet5 import LeNet5

torch.backends.cudnn.benchmark = True
# load_pth = torch.load('lenet_sgd.pth')
load_pth = torch.load('lenet_adam.pth')
torch.cuda.empty_cache()
model = LeNet5().cuda()
model.load_state_dict(load_pth['model'])
name_list = []
layer_list = []
for name, p in model.named_parameters():
    name_list.append(name)
    if len(p.data.size()) == 4 or len(p.data.size()) == 2:
        layer_list.append(name)
optimizer = SLBI_ToolBox(model.parameters(), lr=1e-1, kappa=1, mu=20, weight_decay=0)
optimizer.load_state_dict(load_pth['optimizer'])
optimizer.assign_name(name_list)
optimizer.initialize_slbi(layer_list)

testset = torchvision.datasets.MNIST(root='./data', train=False,download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=128,shuffle=False, num_workers=0)

#### test prune one layer
print('prune conv3')
print('acc before pruning')
evaluate_batch(model, testloader, 'cuda')
print('acc after pruning')
optimizer.prune_layer_by_order_by_name(80, 'conv3.weight', True)
evaluate_batch(model, testloader, 'cuda')
# torch.save(model.state_dict(),'./lenet_sgd_prune.pth')
torch.save(model.state_dict(),'./lenet_adam_prune.pth')
print('acc after recovering')
optimizer.recover()
evaluate_batch(model, testloader, 'cuda')
'''
#### test prune two layers

print('prune conv3 and fc1')
print('acc before pruning')
evaluate_batch(model, test_loader, 'cuda')
print('acc after pruning')
optimizer.prune_layer_by_order_by_list(80, ['conv3.weight', 'fc1.weight'], True)
evaluate_batch(model, test_loader, 'cuda')
print('acc after recovering')
optimizer.recover()
evaluate_batch(model, test_loader, 'cuda')
'''