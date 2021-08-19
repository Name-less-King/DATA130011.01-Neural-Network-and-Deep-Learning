import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
   
class FFM(nn.Moduel):
    def __init__(self,n_hiddens=32):
        super().__init__()
        self.ffm = nn.Sequential(
            ConvBlock(1,n_hiddens,3,1,1),
            ConvBlock(n_hiddens,n_hiddens,3,1,1),
            ConvBlock(n_hiddens,n_hiddens,3,1,1),
            ConvBlock(n_hiddens,n_hiddens,3,1,1),
            ConvBlock(n_hiddens,1,3,1,1)
        )
    #输入低分辨率的图片x，和高分辨率的原图y，进行FFM
    def forward(self,x,y):
        x = F.interpolae(y, size = y.size[2:], mode='bilinear', align_corners=True)

        #经过上采用后并且经过两次残差连接
        y = y.mean(axis=1).unsqueeze(1)
        x = x.mean(axis=1).unsqueeze(1)
        
        x_after_processes = self.ffm(x+y)
        ffm_xy = x_after_processes + y
        
        return ffm_xy





    