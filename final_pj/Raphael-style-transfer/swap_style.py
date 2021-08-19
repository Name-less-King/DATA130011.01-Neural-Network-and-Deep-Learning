from net.tran import FeatureNet, TransformerNet
from net.tran import gram_matrix,batch_normalize
import torch as t
import torchvision as tv
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader
import tqdm
import matplotlib.pyplot as plt
import cv2   
import numpy as np 
from net.swap import style_swap, VGGEncoder, Decoder, TVloss

device=t.device('cuda:2') 
BATCH = 1
EPOCHES = 1000
LR = 1e-4
TV_WEIGHT = 1e-6
PATCH = 3

class TransModel():
    def __init__(self):
        self.encoder = VGGEncoder().to(device)
        self.decoder = Decoder().to(device)
        
        self.optimizer = t.optim.Adam(self.decoder.parameters(), LR)
        self.criterion =  nn.MSELoss()

    def train(self,contentloader,styleloader):

        for epoch in range(EPOCHES):
            for i, (contentdata,styledata) in enumerate(zip(contentloader,styleloader)):

                content,_ = contentdata
                style, _ = styledata

                content = content.to(device)
                style = style.to(device)

                #对content进行灰度化
                
                """
                _,C,_,_ = content.shape 
                if C == 3:
                    content = content.mean(axis=1)
                    content = t.stack((content,content,content),axis=1)
                """

                content_feature = self.encoder(content)
                style_feature = self.encoder(style)

                style_swap_res = []
                for b in range(content_feature.shape[0]):
                    c = content_feature[b].unsqueeze(0)
                    s = style_feature[b].unsqueeze(0)
                    cs = style_swap(c, s, PATCH, 1)
                    style_swap_res.append(cs)
                style_swap_res = t.cat(style_swap_res, 0)

                out_style_swap = self.decoder(style_swap_res)
                out_content = self.decoder(content_feature)
                out_style = self.decoder(style_feature)

                out_style_swap_latent = self.encoder(out_style_swap)
                out_content_latent = self.encoder(out_content)
                out_style_latent = self.encoder(out_style)

                image_reconstruction_loss = self.criterion(content, out_content) + self.criterion(style, out_style)

                feature_reconstruction_loss = self.criterion(style_feature, out_style_latent) +\
                    self.criterion(content_feature, out_content_latent) +\
                    self.criterion(style_swap_res, out_style_swap_latent)

                tv_loss = TVloss(out_style_swap, TV_WEIGHT) + TVloss(out_content, TV_WEIGHT) \
                    + TVloss(out_style, TV_WEIGHT)

                loss = image_reconstruction_loss + feature_reconstruction_loss + tv_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch % 10 == 0:
                print('epoch',epoch,'loss',loss.item())
                plt.figure()
                plt.clf()
                
                out_style_swap = out_style_swap.cpu()[0].permute(1,2,0)
                out_style_swap = out_style_swap.detach().numpy()
                plt.imshow(out_style_swap)
                plt.axis('off')
                plt.savefig('./checkpoints/swap_style/' + str(epoch/10) +'.png')
                plt.close()

img_size = 512
img_mean = [0.485, 0.456, 0.406]
img_std  = [0.229, 0.224, 0.225]

myTransform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Resize(img_size),
        tv.transforms.CenterCrop(img_size),
        #tv.transforms.Normalize(mean=img_mean, std=img_std),
    ])

styleTransform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.RandomCrop(img_size),
        tv.transforms.CenterCrop(img_size),
        #tv.transforms.Normalize(mean=img_mean, std=img_std),
    ])

#训练
contentset = tv.datasets.ImageFolder("/workspace/art/test_images", myTransform)
contentloader = DataLoader(contentset, BATCH, shuffle=True)

styleset = tv.datasets.ImageFolder("/workspace/art/images", styleTransform)
styleloader = DataLoader(styleset, BATCH, shuffle=True)

trans = TransModel()
trans.train(contentloader,styleloader)

#测试
"""
for i in range(1,10):
    content_path ='test_images/img/' + str(i) +'.jpg'
    content_image = tv.datasets.folder.default_loader(content_path)
    content = myTransform(content_image).unsqueeze(0).to(device)
    trans.stylise(style,content,'results/' + str(i) + '.png')
"""