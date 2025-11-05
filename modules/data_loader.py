"""
数据加载模块
提供各种数据集的加载函数
"""
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_data_fashion_mnist(batch_size, resize=None):
    """
    加载Fashion-MNIST数据集
    
    参数:
        batch_size: 批量大小
        resize: 调整图像大小，如果为None则保持原始大小(28x28)
    
    返回:
        train_iter: 训练数据迭代器
        test_iter: 测试数据迭代器
    """
    # 定义数据变换
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    
    # 获取数据目录路径（项目根目录下的 data 文件夹）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data')
    
    # 加载训练集
    mnist_train = datasets.FashionMNIST(
        root=data_dir, train=True, transform=trans, download=True
    )
    # 加载测试集
    mnist_test = datasets.FashionMNIST(
        root=data_dir, train=False, transform=trans, download=True
    )
    
    # 创建数据加载器
    # num_workers=0 避免在macOS上的多进程问题
    # 注意：MPS不支持pin_memory，所以不设置pin_memory
    train_iter = DataLoader(mnist_train, batch_size, shuffle=True, 
                            num_workers=0, persistent_workers=False)
    test_iter = DataLoader(mnist_test, batch_size, shuffle=False, 
                           num_workers=0, persistent_workers=False)
    
    return train_iter, test_iter

