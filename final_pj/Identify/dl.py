import numpy as np
import pickle
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from feature_extract import Tight_frame_classifier

Tight_frame_model = pickle.load(open('model.pkl','rb'))
Feature_index = Tight_frame_model.best_feature()

Train_set = pickle.load(open('Train_set.pkl','rb'))
Test_set = pickle.load(open('Test_set.pkl','rb'))
Train_label = pickle.load(open('Train_label.pkl','rb'))
Test_label = pickle.load(open('Test_label.pkl','rb'))
Disputed = pickle.load(open('tight_frame_D.p','rb'))

Train = Train_set[:,Feature_index]
Test = Test_set[:,Feature_index]
Disputed = Disputed[:,Feature_index]

Train_set = torch.Tensor(Train_set)
Train_label = torch.LongTensor(Train_label)

# 神经网络方法
class Net(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.n_classes = n_classes
        self.relu = nn.ReLU(inplace = True)
        self.sigmoid = nn.Sigmoid(inplace = True)

    def forward(self,x):
        x = nn.linear(x.shape[0],6)
        x = self.sigmoid(x)
        # x = self.relu(x)  
        x = nn.linear(x,1)

        return x

def get_accuracy(model):
    model.eval()
    with torch.no_grad():
        y_pred = model(Test_set)
        _,pred = torch. max(y_pred,dim= 1)

    TP = 0; TN = 0
    for i in range(len(Test_label)):
        if(pred[i] == Test_label[i] == 1):
            TP += 1
    elif(pred[i] == Test_label[i] == 0):
            TN += 1
    TPR = TP / sum(Test_label)
    TNR = TN / (len(Test_label)-sum(Test_label))

    print('\naccuracy: %.2f%%' % (100 * accuracy))
    print('TPR: %.2f%%\nTNR: %.2f%%' % (100 * TPR, 100 * TNR))
    print('precision: %.2f%%\nrecall: %.2f%%' % (100 * precision, 100 * recall))

max_epoch = 100

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9,weight_decay=5e-4)
scheduler = None

model = Net()

for epoch in tqdm(range(max_epoch)):

    model.train()
    running_loss = 0.0

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(Train_set)
    loss = criterion(outputs, Train_label)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()

    if scheduler is not None:
        scheduler.step()

    print('\n Epoch %d Loss: %.3f \n' %(epoch + 1, running_loss / len(Train_set)))
        
    model.eval()
    get_accuracy(model)



