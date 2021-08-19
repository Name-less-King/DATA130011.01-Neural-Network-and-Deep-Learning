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
        self.extracter = FeatureNet().eval().to(device)
        self.lr = 1e-3
        self.optimizer = t.optim.Adam(self.transformer.parameters(), self.lr)

        #需要调整权重，训练次数
        self.content_weight = 1e5
        self.style_weight = 1e11

        #style1 1e10
        #style2 1e11
        self.epoches = 500

    def train(self,dataloader,style):
        with t.no_grad():
            features_style = self.extracter(style)
            gram_style = [gram_matrix(y) for y in features_style]
            #得到channel相似度矩阵

        for epoch in tqdm.tqdm(range(self.epoches)):
            for i, (x, _) in enumerate(dataloader):
                self.optimizer.zero_grad()

                x = x.to(device)
                _,C,_,_ = x.shape 
                
                if C == 3:
                    x = x.mean(axis=1)
                    x = t.stack((x,x,x),axis=1)
                elif C == 1:
                    x = x.squeeze(axis=1)
                    x = t.stack((x,x,x),axis=1)

                #print(x.shape)
                y = self.transformer(x)
                #y = self.transformer(r)

                #y = batch_normalize(y)
                #x = batch_normalize(x)
                features_y = self.extracter(y)
                features_x = self.extracter(x)
            
                #使用relu2的激活值计算内容的损失
                content_loss = self.content_weight * F.mse_loss(features_y.relu2, features_x.relu2)
                
                #风格损失
                gram_y = [gram_matrix(y) for y in features_y]
                style_loss = 0
                for i in range(len(gram_y)):
                    style_loss += F.mse_loss(gram_y[i], gram_style[i].expand_as(gram_y[i]))
                style_loss =  self.style_weight * style_loss 
                
                loss = content_loss + style_loss
                loss.backward()
                self.optimizer.step()

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

class EdgeDetection():
    def __call__(self,img):
        img = np.array(img)
        img = cv2.Canny(img,50,150)
        return img 

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