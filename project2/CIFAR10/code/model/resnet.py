import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, batch_norm=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel) if batch_norm else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.elu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel) if batch_norm else nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel) if batch_norm else nn.Sequential()
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.sigmoid(out)
        # out = self.tanh(out)
        # out = self.elu(out)



        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        # out = self.sigmoid(out)
        # out = self.tanh(out)
        # out = self.elu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, if_bn=True, channel=64):
        super(ResNet18, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel) if if_bn else nn.Sequential(),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(ResBlock(channel, channel, 1, if_bn), ResBlock(channel, channel, 1, if_bn))
        self.layer2 = nn.Sequential(ResBlock(channel, channel*2, 2, if_bn), ResBlock(channel*2, channel*2, 1, if_bn))
        self.layer3 = nn.Sequential(ResBlock(channel*2, channel*4, 2, if_bn), ResBlock(channel*4, channel*4, 1, if_bn))
        self.layer4 = nn.Sequential(ResBlock(channel*4, channel*8, 2, if_bn), ResBlock(channel*8, channel*8, 1, if_bn))

        self.fc = nn.Linear(channel*8, num_classes)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x

class PlainNet(nn.Module):
    def __init__(self, num_classes=10, if_bn=True):
        super(PlainNet, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64) if if_bn else nn.Sequential(),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x

class ResNet18GAP(nn.Module):
    def __init__(self, num_classes=10, if_bn=True):
        super(ResNet18GAP, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64) if if_bn else nn.Sequential(),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(ResBlock(64, 64, 1, if_bn), ResBlock(64, 64, 1, if_bn))
        self.layer2 = nn.Sequential(ResBlock(64, 128, 2, if_bn), ResBlock(128, 128, 1, if_bn))
        self.layer3 = nn.Sequential(ResBlock(128, 256, 2, if_bn), ResBlock(256, 256, 1, if_bn))
        self.layer4 = nn.Sequential(ResBlock(256, 512, 2, if_bn), ResBlock(512, 512, 1, if_bn))

        self.layer5 = nn.Conv2d(512, 10, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.layer5(x)
        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, start_dim=1)

        return x
