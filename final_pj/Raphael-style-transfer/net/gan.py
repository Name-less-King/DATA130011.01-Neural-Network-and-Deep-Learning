import torch
from torch import nn
from torch.nn import functional as F

# for GAN, REF is also my github: https://github.com/TrueNobility303/GAN-face-generator

# implement of cycle GAN
# 也可视为一种跨域操作，使用GAN从A域跨到B域

#使用instance norm代替BN
def Conv(n_input, n_output, k_size=4, stride=2, padding=0, bn=False):
    return nn.Sequential(
        nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),
        nn.InstanceNorm2d(n_output,affine=True),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(p=0.2, inplace=False))

def Deconv(n_input, n_output, k_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(n_input, n_output,kernel_size=k_size,stride=stride, padding=padding,bias=False),
        nn.InstanceNorm2d(n_output,affine=True),
        nn.ReLU(inplace=True))

class Discriminator(nn.Module):
    def __init__(self, nc=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, nc,kernel_size=4,stride=2,padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            Conv(nc, nc*2, 4,2,1),
            Conv(nc*2, nc*4, 4,2,1),
            Conv(nc*4, nc*8, 4,2,1),
            nn.Conv2d(nc*8, 1,4,1,0, bias=False),
            nn.Sigmoid())
        
    def forward(self, input):
        #print(input.shape)
        return self.net(input)


