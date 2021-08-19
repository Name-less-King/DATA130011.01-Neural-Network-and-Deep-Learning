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
from utils import *

# ## Constants (parameters) initialization
device_id = 0
num_workers = 0
batch_size = 128

# add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

'''
I won't use these codes:   

device_id = device_id
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device("cuda:{}".format(3) if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(3))
'''

# Make sure you are using the right device.
device = torch.device("cuda:0")


# Initialize your data loader and
# make sure that dataloader works
# as expected by observing one
# sample from it.
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)


# This function is used to calculate the accuracy of model classification
# one for train, one for val
def get_train_accuracy(model):
    size = 0
    correct = 0
    with torch.no_grad():
        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            _, y_pred = torch.max(outputs.data, 1)
            size += y.size(0)
            correct += (y_pred == y).sum().item()

    print('\n Train Accuracy : %.2f %% \n ' % (100 * correct / size))
    return(correct / size)
    
def get_val_accuracy(model):
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

    print('\n Val Accuracy : %.2f %% \n' % (100 * correct / size))
    return(correct / size)

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


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    if type(model) == VGG_A:
        layer_id = 3
    if type(model) == VGG_A_BatchNorm:
        layer_id = 4
    else:
        pass

    model.to(device)
    
    learning_curve = []
    train_accuracy_curve = []
    val_accuracy_curve = [] 
    diff = []

    batches_n = len(train_loader)
    
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        aver_loss = 0.0;
        
        for batch,data in enumerate(train_loader):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            
            aver_loss += loss
            
            loss.backward()
            optimizer.step()
            
            current_weights = get_vgg_weights(model)
            grad = model.features[layer_id].weight.grad.clone()
            
            
            # perform optimizer's step
            optimizer.step()
            new_weights = get_vgg_weights(model)
            

            # compute the gradients of Layer5
            set_vgg_weights(model, current_weights)
            set_vgg_weights(model, new_weights, feature_border=layer_id)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            grad_new = model.features[layer_id].weight.grad.clone()
            
            # set all weights to new
            set_vgg_weights(model, new_weights)
            
            # calculate ICS results
            diff.append(torch.norm(grad - grad_new, 2))
            

        
        aver_loss = aver_loss/batches_n
        learning_curve.append(aver_loss)
        
        model.eval()
        train_accuracy_curve.append(get_train_accuracy(model))
        val_accuracy_curve.append(get_val_accuracy(model))
    
    return learning_curve,train_accuracy_curve,val_accuracy_curve,diff

if __name__ == '__main__':
    # Train your model
    # feel free to modify
    epo = 20
    #loss_save_path = ''
    #grad_save_path = ''

    set_random_seeds(seed_value=2020, device=device)
    model = VGG_A()
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()
    _,_,_,diff = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
    
    
    set_random_seeds(seed_value=2020, device=device)
    model = VGG_A_BatchNorm()
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()
    _,_,_,diff1 = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
    
    plt.style.use('ggplot')
    # plt.plot([i for i in range(1, epo + 1)], vac, c = 'green', label = 'Standard VGG')
    # plt.plot([i for i in range(1, epo + 1)], vac1, c = 'firebrick', label = 'Standard VGG + BatchNorm')    

    # plt.xticks(np.arange(1, epo+1, 2))

    plt.plot([i for i in range(1, len(diff) + 1)], diff, c = 'green', label = 'Standard VGG')
    plt.plot([i for i in range(1, len(diff1) + 1)], diff1, c = 'firebrick', label = 'Standard VGG + BatchNorm')    

    plt.xticks(np.arange(1, len(diff)+1, 1000))
    plt.xlabel('Step')
    plt.ylabel('L2-diff')
    plt.title('Layer #5')
    plt.legend(loc='best', fontsize='x-large')
    plt.savefig('./Layer #5')