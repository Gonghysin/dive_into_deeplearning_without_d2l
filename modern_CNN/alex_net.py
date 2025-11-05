import sys
import os
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import dropout, nn
from torch.nn.functional import pad
from modules import load_data_fashion_mnist, train_ch6

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

    # 加载数据（会自动根据平台配置DataLoader）
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224, auto_config=True)

    lr, num_epochs = 0.01, 10

    # 开始训练（会自动检测并使用最佳设备：Windows CUDA / Mac MPS / CPU）
    train_ch6(net, train_iter, test_iter, num_epochs, lr, device=None)
