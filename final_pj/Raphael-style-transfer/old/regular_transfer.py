from net.tran import FeatureNet, TransformerNet
from net.tran import gram_matrix,batch_normalize
import torch as t
import torchvision as tv
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import tqdm
import matplotlib.pyplot as plt  

device=t.device('cuda:2')
BATCH = 8

def get_kernel():
    kernel = nn.Conv2d(3,3,3,1,1,bias=False)
    gradient_kernel = t.Tensor([[0,-1,0],[-1,0,1],[0,1,0]])
    gradient_kernel = gradient_kernel.view(1,1,3,3)
    gradient_kernel = gradient_kernel.expand_as(kernel.weight)
    kernel.weight.data = gradient_kernel.data 
    return kernel

class TransModel():
    def __init__(self):
        self.transformer = TransformerNet().to(device)
        self.extracter = FeatureNet().eval().to(device)
        self.lr = 1e-3
        self.optimizer = t.optim.Adam(self.transformer.parameters(), self.lr)

        #需要调整权重，训练次数
        self.content_weight = 1e5
        self.style_weight = 5e10
        self.regular_weight = 1e2

        #style1 1e10
        #style2 1e11
        self.epoches = 10

        #卷积核提取边缘特征
        self.kernel = get_kernel().to(device)

    def train(self,dataloader,style):
        with t.no_grad():
            features_style = self.extracter(style)
            gram_style = [gram_matrix(y) for y in features_style]
            #得到channel相似度矩阵

        for epoch in range(self.epoches):
            for i, (x, _) in tqdm.tqdm(enumerate(dataloader)):
                
                self.optimizer.zero_grad()

                x = x.to(device)
                y = self.transformer(x)

                if i > 100:
                    break
                
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
                
                regular_loss = self.regular_weight * t.norm(self.kernel(y))

                loss = content_loss + style_loss + regular_loss
                loss.backward()
                self.optimizer.step()

            if epoch % 1 == 0:
                plt.figure()
                
                origin_img = x.data.cpu()[1].permute(1,2,0)
                style_img = style.cpu()[0].permute(1,2,0)
                new_img  = y.data.cpu()[1].permute(1,2,0)

                plt.subplot(131)
                plt.imshow(origin_img)
                plt.xticks([]),plt.yticks([])
                plt.subplot(132)
                plt.imshow(style_img)
                plt.xticks([]),plt.yticks([])
                plt.subplot(133)
                plt.imshow(new_img)
                plt.xticks([]),plt.yticks([])

                #中间结果存放于dump下
                plt.savefig('./dump/' + str(epoch) +'.png')
                plt.close()

    def stylise(self,style,content,save_path):
        plt.figure()
            
        origin_img = content.cpu()[0].permute(1,2,0)
        style_img = style.cpu()[0].permute(1,2,0)

        y = self.transformer(content) 
        new_img  = y.data.cpu()[0].permute(1,2,0)
        
        plt.subplot(131)
        plt.imshow(origin_img)
        plt.xticks([]),plt.yticks([])
        plt.subplot(132)
        plt.imshow(style_img)
        plt.xticks([]),plt.yticks([])
        plt.subplot(133)
        plt.imshow(new_img)
        plt.xticks([]),plt.yticks([])

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

styleTransform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.RandomCrop(img_size),
        #tv.transforms.Normalize(mean=img_mean, std=img_std),
    ])
#使用tiny-imagenet的测试集作为数据集,共10000张图片
dataset = tv.datasets.ImageFolder("/datasets/tiny-imagenet/test", myTransform)
dataloader = DataLoader(dataset, BATCH)

style_path = 'images/Raphael_Project_image/7. 243x413 .tiff'
style_image = tv.datasets.folder.default_loader(style_path)
style = myTransform(style_image).unsqueeze(0).to(device)
#print('style',style.shape)

trans = TransModel()
trans.train(dataloader,style)

for i in range(1,10):
    content_path ='test_images/img/' + str(i) +'.jpg'
    content_image = tv.datasets.folder.default_loader(content_path)
    content = myTransform(content_image).unsqueeze(0).to(device)
    trans.stylise(style,content,'results/' + str(i) + '.png')