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
from vgg import VGG_A_BatchNorm as VGG

torch.backends.cudnn.benchmark = True
load_pth = torch.load('vgg_sgd.pth')
# load_pth = torch.load('vgg_adam.pth')
torch.cuda.empty_cache()
model = VGG().cuda()
model.load_state_dict(load_pth['model'])
name_list = []
layer_list = []
for name, p in model.named_parameters():
    name_list.append(name)
    if len(p.data.size()) == 4 or len(p.data.size()) == 2:
        layer_list.append(name)
optimizer = SLBI_ToolBox(model.parameters(), lr=1e-1, kappa=1, mu=500, weight_decay=5e-4)
# optimizer = SLBI_ToolBox(model.parameters(), lr=3e-4, kappa=5, mu=100, betas=(0.9,0.999), eps = 1e-8, weight_decay=5e-4)
optimizer.load_state_dict(load_pth['optimizer'])
optimizer.assign_name(name_list)
optimizer.initialize_slbi(layer_list)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,shuffle=False, num_workers=0)

#### test prune one layer
print('prune Layer #5')
print('acc before pruning')
evaluate_batch(model, testloader, 'cuda')
print('acc after pruning')
optimizer.prune_layer_by_order_by_name(30, 'features.4.weight', True)
evaluate_batch(model, testloader, 'cuda')
torch.save(model.state_dict(),'./vgg_sgd_prune.pth')
#torch.save(model.state_dict(),'./vgg_adam_prune.pth')
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