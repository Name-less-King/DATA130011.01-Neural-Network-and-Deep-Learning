import torch
import torch.nn as nn 
from net.tran import FeatureNet, TransformerNet
from net.tran import gram_matrix,batch_normalize
import torch as t
import torchvision as tv
from torch.nn import functional as F
from torch.utils.data import DataLoader
import tqdm
import matplotlib.pyplot as plt  

device=t.device('cuda:3') if t.cuda.is_available() else t.device('cpu')
BATCH = 8

class TransModel():
    def __init__(self):
        self.transformer = TransformerNet().to(device)
        self.extracter = FeatureNet().eval().to(device)
        self.recover = TransformerNet().to(device)
        
        #调整学习率
        self.optimizer_tran = t.optim.Adam(self.transformer.parameters(), 1e-3)
        self.optimizer_recover = t.optim.Adam(self.recover.parameters(), 3e-4)

        #需要调整权重，训练次数
        self.content_weight = 1e5
        self.style_weight = 5e10
        self.recover_weight = 1
        self.epoches = 100

    def train(self,dataloader,style):
        with t.no_grad():
            features_style = self.extracter(style)
            gram_style = [gram_matrix(y) for y in features_style]
            #得到channel相似度矩阵

        for epoch in range(self.epoches):
            self.transformer.train()
            self.recover.train()
            tot_recover_loss = 0
            for i, (x, _) in enumerate(dataloader):
                if i > 100:
                    break
                
                #训练转换网络,x to y

                self.optimizer_tran.zero_grad()

                x = x.to(device)
                y = self.transformer(x)
                features_y = self.extracter(y)
                features_x = self.extracter(x)
                
                content_loss = self.content_weight * F.mse_loss(features_y.relu2, features_x.relu2)

                #风格损失
                gram_y = [gram_matrix(ay) for ay in features_y]
                style_loss = 0
                for i in range(len(gram_y)):
                    style_loss += F.mse_loss(gram_y[i], gram_style[i].expand_as(gram_y[i]))
                style_loss =  self.style_weight * style_loss 
                
                loss = style_loss + content_loss
                loss.backward()
                self.optimizer_tran.step()

                #训练复原网络

                y = y.clone().detach()
                self.optimizer_recover.zero_grad()
                z = self.recover(y)
                recover_loss = self.recover_weight * F.mse_loss(x,z)
                recover_loss.backward()
                self.optimizer_recover.step()
                tot_recover_loss += recover_loss.item()

            self.transformer.eval()
            self.recover.eval()

            if epoch % 1 == 0:
                print('epoch',epoch,'recover loss',tot_recover_loss)
                plt.figure()
                x = x.to(device)
                y = self.transformer(x)
                z = self.recover(y)
                
                origin_img = x.data.cpu()[1].permute(1,2,0)
                style_img = style.cpu()[0].permute(1,2,0)
                new_img  = y.data.cpu()[1].permute(1,2,0)
                recover_img = z.data.cpu()[1].permute(1,2,0)

                H = origin_img.shape[0]
                W = origin_img.shape[1]
                plt.subplot(141)
                gray_origin_img = torch.zeros(H,W,1)
                gray_origin_img[:,:,0] =  0.212674 * origin_img[:,:,0] + 0.715160 * origin_img[:,:,1] + 0.072169 * origin_img[:,:,2]
                plt.imshow(origin_img,cmap='gray')
                plt.xticks([]),plt.yticks([])
                plt.title('content')

                plt.subplot(142)
                plt.imshow(style_img)
                plt.xticks([]),plt.yticks([])
                plt.title('style')

                plt.subplot(143)
                plt.imshow(new_img)
                plt.xticks([]),plt.yticks([])
                plt.title('generate')
                plt.subplot(144)
               
                plt.subplot(144)
                gray_recover_img = torch.zeros(H,W,1)
                gray_recover_img =  0.212674 * recover_img[:,:,0] + 0.715160 * recover_img[:,:,1] + 0.072169 * recover_img[:,:,2]
                plt.imshow(recover_img,cmap='gray')
                plt.xticks([]),plt.yticks([])
                plt.title('recover')

                #中间结果存放于dump下
                plt.savefig('./dump/' + str(epoch) +'.png')
                plt.close()

    def stylise(self,style,content,save_path):
        plt.figure()
            
        origin_img = content.cpu()[0].permute(1,2,0)
        style_img = style.cpu()[0].permute(1,2,0)

        y = self.transformer(content) 
        new_img  = y.data.cpu()[0].permute(1,2,0)
        z = self.recover(y)
        recover_img = z.data.cpu()[0].permute(1,2,0)
        
        H = origin_img.shape[0]
        W = origin_img.shape[1]
        plt.subplot(141)
        gray_origin_img = torch.zeros(H,W,1)
        gray_origin_img[:,:,0] =  0.212674 * origin_img[:,:,0] + 0.715160 * origin_img[:,:,1] + 0.072169 * origin_img[:,:,2]
        plt.imshow(origin_img,cmap='gray')
        plt.xticks([]),plt.yticks([])
        plt.title('content')

        plt.subplot(142)
        plt.imshow(style_img)
        plt.xticks([]),plt.yticks([])
        plt.title('style')

        plt.subplot(143)
        plt.imshow(new_img)
        plt.xticks([]),plt.yticks([])
        plt.title('generate')
        plt.subplot(144)
        
        plt.subplot(144)
        gray_recover_img = torch.zeros(H,W,1)
        gray_recover_img =  0.212674 * recover_img[:,:,0] + 0.715160 * recover_img[:,:,1] + 0.072169 * recover_img[:,:,2]
        plt.imshow(recover_img,cmap='gray')
        plt.xticks([]),plt.yticks([])
        plt.title('recover')

        plt.savefig(save_path)
        plt.close()


img_size = 256
img_mean = [0.485, 0.456, 0.406]
img_std  = [0.229, 0.224, 0.225]
myTransform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Resize(img_size),
        tv.transforms.CenterCrop(img_size),
        #tv.transforms.Normalize(mean=img_mean, std=img_std),
    ])

#使用tiny-imagenet的测试集作为数据集,共10000张图片
dataset = tv.datasets.ImageFolder("/datasets/tiny-imagenet/test", myTransform)
dataloader = DataLoader(dataset, BATCH)

style_path = 'images/Raphael_Project_image/3A Luca Giordano after Raphael Br Mus.jpg'
style_image = tv.datasets.folder.default_loader(style_path)
style = myTransform(style_image).unsqueeze(0).to(device)

trans = TransModel()
trans.train(dataloader,style)

for i in range(1,9):
    content_path ='test_images/img/' + str(i) +'.jpg'
    content_image = tv.datasets.folder.default_loader(content_path)
    content = myTransform(content_image).unsqueeze(0).to(device)
    trans.stylise(style,content,'results_cycle/' + str(i) + '.png')