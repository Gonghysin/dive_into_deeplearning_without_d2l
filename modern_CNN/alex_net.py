import sys
import os
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import dropout, nn
from torch.nn.functional import pad
from modules.data_loader import load_data_fashion_mnist
from modules.trainer import train_ch6, try_gpu

net = nn.Sequential(
    # 这里使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(1,96,kernel_size=11,stride=4,padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),

    nn.Linear(6400,4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096,4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096,10)
)

if __name__ == '__main__':
    print("---")
    print(net)
    print("---")

    X = torch.randn(1, 1, 224, 224)
    for layer in net:
        X=layer(X)
        print(layer.__class__.__name__,'output shape:\t',X.shape)

    batch_size = 128

    # 使用示例
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

    lr, num_epochs = 0.01, 10

    # 开始训练
    train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())
