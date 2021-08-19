import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
from resnet import ResNet18

batchsize = 128
max_epoch = 240


# set random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# use GPU or CPU
device = torch.device('cuda:0')

class Cutout:
    def __init__(self, size=16, p=0.5):
        self.size = size
        self.half_size = size // 2
        self.p = p

    def __call__(self, image):
        if torch.rand([1]).item() > self.p:
            return image

        left = torch.randint(-self.half_size, image.size(1) - self.half_size, [1]).item()
        top = torch.randint(-self.half_size, image.size(2) - self.half_size, [1]).item()
        right = min(image.size(1), left + self.size)
        bottom = min(image.size(2), top + self.size)

        image[:, max(0, left): right, max(0, top): bottom] = 0
        return image

# transformation of input data
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
    Cutout()
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# define classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def get_train_accuracy(model):
    size = 0
    correct = 0
    with torch.no_grad():
        for data in trainloader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            _, y_pred = torch.max(outputs.data, 1)
            size += y.size(0)
            correct += (y_pred == y).sum().item()

    print('\n Train Accuracy : %.2f %% \n ' % (100 * correct / size))
    return(correct / size)
    
def get_val_accuracy(model):
    size = 0
    correct = 0
    with torch.no_grad():
        for data in testloader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            _, y_pred = torch.max(outputs.data, 1)
            size += y.size(0)
            correct += (y_pred == y).sum().item()

    print('\n Val Accuracy : %.2f %% \n' % (100 * correct / size))
    return(correct / size)


if __name__ == '__main__':

    setup_seed(0)

    """Load in Data"""

    time_start = time.time()
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize,shuffle=False, num_workers=0)
    print('Totally Load in Time Cost', time.time() - time_start,'\n')


    """Network Parameters"""

    net = ResNet18().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9,weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [135,185], gamma=0.1)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100,140], gamma=0.1)
    """Train the Network"""

    # loop over the dataset multiple times
    time_start = time.time()
    
    for epoch in range(max_epoch):

        net.train()
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if scheduler is not None:
            scheduler.step()
        
        if (epoch+1) % 5 ==0:
            # print statistics every five epoch
            print('\n Epoch %d Loss: %.3f \n' %(epoch + 1, running_loss / len(trainloader)))
        
            net.eval()
            get_train_accuracy(net)
            get_val_accuracy(net)

    print('Finished Training! Totally Training Time Cost',
          time.time() - time_start, '\n')

    # save the model
    PATH = './resnetsch-240.pth'
    torch.save(net.state_dict(), PATH)


    """Evaluate Performance"""

    net = ResNet18()
    net.load_state_dict(torch.load(PATH))
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('\n Accuracy of the network on the 10000 test images: %.2f %%'
          % (100 * correct / total))





