import torch
from torch import nn

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))

if __name__ == '__main__':
    print(net)
    X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__,'output shape: \t',X.shape)

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from modules import load_data_fashion_mnist, train_ch6
    lr, num_epochs, batch_size = 0.9, 10, 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size,auto_config=True)
    train_ch6(net, train_iter, test_iter, num_epochs, lr, device=None)

    # 训练完成！最终测试准确率: 0.8259