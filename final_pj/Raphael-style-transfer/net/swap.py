import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
import torch 

class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        self.slice1 = vgg[: 2]
        self.slice2 = vgg[2: 7]
        self.slice3 = vgg[7: 12]
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, images, retrun_hidden_features=False):
        h1 = self.slice1(images)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        if retrun_hidden_features is False:
            return h3
        else: 
            return h1,h2,h3

class RC(nn.Module):
    """A wrapper of ReflectionPad2d and Conv2d"""
    def __init__(self, in_channels, out_channels, kernel_size=3, pad_size=1, activated=True):
        super().__init__()
        self.pad = nn.ReflectionPad2d((pad_size, pad_size, pad_size, pad_size))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.activated = activated

    def forward(self, x):
        h = self.pad(x)
        h = self.conv(h)
        if self.activated:
            return F.relu(h)
        else:
            return h

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rc1 = RC(256, 128, 3, 1)
        self.rc2 = RC(128, 128, 3, 1)
        self.rc3 = RC(128, 64, 3, 1)
        self.rc4 = RC(64, 64, 3, 1)
        self.rc5 = RC(64, 3, 3, 1, False)

    def forward(self, features):
        h = self.rc1(features)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc2(h)
        h = self.rc3(h)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc4(h)
        h = self.rc5(h)
        return torch.sigmoid(h)

def style_swap(content_feature, style_feature, kernel_size, stride=1):
    kh, kw = kernel_size, kernel_size
    sh, sw = stride, stride

    patches = style_feature.unfold(2, kh, sh).unfold(3, kw, sw)

    patches = patches.permute(0, 2, 3, 1, 4, 5)
    patches = patches.reshape(-1, *patches.shape[-3:]) # (patch_numbers, C, kh, kw)

    norm = torch.norm(patches.reshape(patches.shape[0], -1), dim=1).reshape(-1, 1, 1, 1)

    noramalized_patches = patches / norm

    conv_out = F.conv2d(content_feature, noramalized_patches)

    one_hots = torch.zeros_like(conv_out)
    one_hots.scatter_(1, conv_out.argmax(dim=1, keepdim=True), 1)

    deconv_out = F.conv_transpose2d(one_hots, patches)

    overlap = F.conv_transpose2d(one_hots, torch.ones_like(patches))

    res = deconv_out / overlap
    return res

def TVloss(img, tv_weight):
    w_variance = torch.sum(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
    h_variance = torch.sum(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss
