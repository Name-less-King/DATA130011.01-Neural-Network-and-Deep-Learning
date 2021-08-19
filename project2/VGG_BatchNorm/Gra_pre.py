import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
lrs =  [ 1e-2, 5e-3, 1e-3]
epo = 20
device_id = 0
num_workers = 0
batch_size = 128

# add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')


# Make sure you are using the right device.
device = torch.device("cuda:0")


# Initialize your data loader and
# make sure that dataloader works
# as expected by observing one
# sample from it.
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)



# This function is used to calculate the accuracy of model classification
def get_accuracy(model):
    size = 0
    correct = 0
    with torch.no_grad():
        for data in val_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            _, y_pred = torch.max(outputs.data, 1)
            size += y.size(0)
            correct += (y_pred == y).sum().item()

    print('Accuracy : %.2f %%' % (100 * correct / size))

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# get the gradient distance
def VGG_Grad_Pred(model, pre_grad):
    
    if type(model) == VGG_A:
        f_index = [0, 3, 6, 8, 11, 13, 16, 18]
        c_index = [0, 2, 4]
    if type(model) == VGG_A_BatchNorm:
        f_index = [0, 4, 8, 11, 15, 18, 22, 25]
        c_index = [0, 3, 6]
    else:
        pass

    # get the gradients
    grad = torch.tensor([]).to(device)
    for i in f_index:
        grad = torch.cat((grad, model.features[i].weight.grad.view(-1)))
    for i in c_index:
        grad = torch.cat((grad, model.classifier[i].weight.grad.view(-1)))

    # return l2-distance
    if len(pre_grad)==0:
        return np.nan, grad
    else:
        dist = torch.norm(grad-pre_grad)
        return dist, grad

# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    
    dists = []
    grad = torch.tensor([])
    
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()
            
            dist,grad = VGG_Grad_Pred(model,grad)
            dists.append(dist)
            
    return dists

def plot_loss_landscape(iteration,VGG_A_max_curve,
                        VGG_A_min_curve,VGG_A_BN_max_curve,
                        VGG_A_BN_min_curve):
    
    plt.style.use('ggplot')
    
    # plot VGG_A curve
    plt.plot(iteration, VGG_A_max_curve, c='green')
    plt.plot(iteration, VGG_A_min_curve, c='green')
    plt.fill_between(iteration, VGG_A_max_curve, 
                     VGG_A_min_curve, color='lightgreen', 
                     label='Standard VGG')

    # plot VGG_A_BatchNorm  curve
    plt.plot(iteration, VGG_A_BN_max_curve, c='firebrick')
    plt.plot(iteration, VGG_A_BN_min_curve, c='firebrick')
    plt.fill_between(iteration, VGG_A_BN_max_curve, 
                     VGG_A_BN_min_curve, color='lightcoral',
                     label='Standard VGG + BatchNorm')
    
    
    plt.xticks(np.arange(0, iteration[-1], 1000))
    plt.xlabel('Steps')
    plt.ylabel('Gradient distance')
    plt.title('Gradient distance Landscape')
    plt.legend(loc='upper right', fontsize='x-large')
    plt.savefig('T2.4.1 VGG_Cmp_BN.png')

if __name__ == '__main__':
    # Train your model
    # feel free to modify
    #loss_save_path = ''
    #grad_save_path = ''

    set_random_seeds(seed_value=2020, device=device)
    
    VGG_A_dists = []
    VGG_A_BN_dists = []
    
    for lr in lrs:
        model = VGG_A()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        VGG_A_dists.append(train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo))
        
        model = VGG_A_BatchNorm()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        VGG_A_BN_dists.append(train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo))
    
    
    VGG_A_dists = np.array(VGG_A_dists)
    VGG_A_BN_dists = np.array(VGG_A_BN_dists)

    iteration = []
    VGG_A_min_curve = []
    VGG_A_max_curve = []
    VGG_A_BN_min_curve = []
    VGG_A_BN_max_curve = []
    
    VGG_A_min = VGG_A_dists.min(axis=0).astype(float)
    VGG_A_max = VGG_A_dists.max(axis=0).astype(float)
    VGG_A_BN_min = VGG_A_BN_dists.min(axis=0).astype(float)
    VGG_A_BN_max = VGG_A_BN_dists.max(axis=0).astype(float)
    for i in range(len(VGG_A_min)):
        if i%30 == 0:
            VGG_A_min_curve.append(VGG_A_min[i])
            VGG_A_max_curve.append(VGG_A_max[i])
            VGG_A_BN_min_curve.append(VGG_A_BN_min[i])
            VGG_A_BN_max_curve.append(VGG_A_BN_max[i])
            iteration.append(i)
    
    plot_loss_landscape(iteration,VGG_A_max_curve,
                        VGG_A_min_curve,VGG_A_BN_max_curve,
                        VGG_A_BN_min_curve)
    
    