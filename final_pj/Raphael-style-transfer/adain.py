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
from net.adain import AdaIN, calc_mean_std

device=t.device('cuda:2') 
BATCH = 1
EPOCHES = 200
LR = 1e-5

LAM = 10
ALPHA = 1

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

                content_features = self.encoder(content,retrun_hidden_features=True)
                style_features = self.encoder(style,retrun_hidden_features=True)

                #使用AdaIN将content迁移至style
                adain_features = AdaIN(content_features[-1], style_features[-1])
                adain_features = ALPHA * adain_features + (1 - ALPHA) * content_features[-1]
                out = self.decoder(adain_features)
                
                output_features = self.encoder(out,retrun_hidden_features=True)
                
                content_loss = self.criterion(content_features[-1], output_features[-1])

                style_loss = 0
                for c,s in zip(output_features,style_features):
                    content_mean, content_std = calc_mean_std(c)
                    style_mean, style_std = calc_mean_std(s)
                    style_loss += self.criterion(content_mean,style_mean) + self.criterion(content_std,style_std)

                loss = content_loss + LAM * style_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch % 10 == 0:
                print('epoch',epoch,'loss',loss.item())
                plt.figure()
                plt.clf()
                
                out = out.cpu()[0].permute(1,2,0)
                out = out.detach().numpy()
                plt.imshow(out)
                plt.axis('off')
                plt.savefig('./checkpoints/adain/' + str(epoch/10) +'.png')
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