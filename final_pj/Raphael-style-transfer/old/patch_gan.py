from net.tran import FeatureNet, TransformerNet
from net.tran import gram_matrix,batch_normalize
import torch as t
import torchvision as tv
from torch.nn import functional as F
from torch.utils.data import DataLoader
import tqdm
import matplotlib.pyplot as plt
import cv2   
import numpy as np 

device=t.device('cuda:0') if t.cuda.is_available() else t.device('cpu')
BATCH = 1

class TransModel():
    def __init__(self):
        self.transformer = TransformerNet().to(device)
        self.discriminator = TransformerNet().to(device)

        self.extracter = FeatureNet().eval().to(device)

        self.trans_lr = 2e-4
        self.dis_lr = 1e-3

        self.content_weight = 1
        self.style_weight = 1
        self.dis_weight = 1

        self.optimizer_trans = t.optim.Adam(self.transformer.parameters(), self.trans_lr)
        self.optimizer_dis = t.optim.Adam(self.discriminator.parameters(), self.dis_lr)

        self.epoches = 500

    def train(self,dataloader,style):

        for epoch in range(self.epoches):
            for i, (x, _) in enumerate(dataloader):
                #预处理x
                x = x.to(device)
                _,C,_,_ = x.shape 
                if C == 3:
                    x = x.mean(axis=1)
                    x = t.stack((x,x,x),axis=1)
                elif C == 1:
                    x = x.squeeze(axis=1)
                    x = t.stack((x,x,x),axis=1)

                #训练判别器
                y = self.transformer(x)
                real_loss = 1 - self.discriminator(style).mean()
                fake_loss = self.discriminator(y).mean()
                dis_loss = self.dis_weight * (real_loss+fake_loss)
                self.optimizer_dis.zero_grad()
                dis_loss.backward()
                self.optimizer_dis.step()
                
                #训练生成器
                y = self.transformer(x)

                #生成器的内容损失
                features_y = self.extracter(y)
                features_x = self.extracter(x)
                content_loss = self.content_weight * F.mse_loss(features_y.relu2, features_x.relu2)
                
                #生成器的风格损失，生成器的任务是欺骗判别器
                style_loss = 1 - self.discriminator(y).mean()
                style_loss =  self.style_weight * style_loss 
                
                trans_loss = content_loss + style_loss
                self.optimizer_trans.zero_grad()
                trans_loss.backward()
                self.optimizer_trans.step()

            print('epoch',epoch,'dis_loss',dis_loss.item(),'content_loss',content_loss.item(),'style_loss',style_loss.item())
            if epoch % 10 == 0:
                plt.figure()
                
                origin_img = x.data.cpu()[0].permute(1,2,0)
                style_img = style.cpu()[0].permute(1,2,0)
                new_img  = y.data.cpu()[0].permute(1,2,0)

                plt.subplot(111)
                plt.imshow(new_img)
                plt.xticks([]),plt.yticks([])

                #中间结果存放于dump下
                plt.savefig('./dump/' + str(epoch/10) +'.png')
                plt.close()


    def stylise(self,style,content,save_path):
        plt.figure()
            
        origin_img = content.cpu()[0].permute(1,2,0)
        style_img = style.cpu()[0].permute(1,2,0)

        y = self.transformer(content) 
        new_img  = y.data.cpu()[0].permute(1,2,0)
        
        plt.subplot(111)
        plt.imshow(new_img)
        plt.xticks([]),plt.yticks([])

        plt.savefig(save_path)
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

#使用end2end的训练
dataset = tv.datasets.ImageFolder("/workspace/art/test_images", myTransform)
dataloader = DataLoader(dataset, BATCH)

style_path = 'images/Raphael_Project_image/6. 197x168.tiff'
style_image = tv.datasets.folder.default_loader(style_path)
style = styleTransform(style_image).unsqueeze(0).to(device)

trans = TransModel()
trans.train(dataloader,style)

for i in range(1,10):
    content_path ='test_images/img/' + str(i) +'.jpg'
    content_image = tv.datasets.folder.default_loader(content_path)
    content = myTransform(content_image).unsqueeze(0).to(device)
    trans.stylise(style,content,'results/' + str(i) + '.png')