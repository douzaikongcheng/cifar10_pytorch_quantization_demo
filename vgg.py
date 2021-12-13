import torch
import torch.nn as nn
from torch.autograd import Variable


class vgg(nn.Module):
    def __init__(self):
        super(vgg, self).__init__()
        self.feature1 = nn.Sequential(nn.Conv2d(3, 64, 3, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.down1 = nn.MaxPool2d(2)
        self.feature2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.down2 = nn.MaxPool2d(2)
        self.feature3 = nn.Sequential(nn.Conv2d(64, 64, 3, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.down3 = nn.MaxPool2d(2)
        self.adp = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        x = self.feature1(x)
        x = self.down1(x)
        x = self.feature2(x)
        x = self.down2(x)
        x = self.feature3(x)
        x = self.down3(x)
        x = self.adp(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y


if __name__ == '__main__':
    net = vgg()
    x = Variable(torch.FloatTensor(1, 3, 32, 32))
    y = net(x)
    print(y.data.shape)
