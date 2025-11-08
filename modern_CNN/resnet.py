"""
ResNet 实现
演示使用 @save 装饰器自动注册函数/类到 modules 模块
"""
import torch
from torch import nn
from torch.nn import functional as F

# 导入 save 装饰器

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules import save , load_data_fashion_mnist , train_ch6

# 使用 @save 装饰器注册 Residual 类到 modules 模块
@save(category="models")
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, 
        use_1x1conv=False, strides=1):
        super().__init__()    
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X # 将输入跳过两个卷积层直接加在最后的ReLU激活函数前
        return F.relu(Y)

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5,
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(), nn.Linear(512, 10))

if __name__ == '__main__':
    blk = Residual(3,3)
    X = torch.rand(4, 3, 6, 6)
    Y = blk(X)
    print(blk)
    print(Y.shape)
    blk = Residual(3,6, use_1x1conv=True, strides=2)
    print(blk(X).shape)    

    print('=' * 60)
    print(net)
    print('=' * 60)
    X = torch.rand(1, 1, 224, 224)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__,'output shape:\t',X.shape)

    lr, num_epochs, batch_size = 0.05, 10, 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size, auto_config=True)
    # train_ch6(net, train_iter, test_iter, num_epochs, lr, device=None)
