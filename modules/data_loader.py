"""
数据加载模块
提供各种数据集的加载函数
"""
import os
import platform
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloader_config():
    """
    根据系统平台和可用设备自动配置DataLoader参数
    
    返回:
        dict: 包含num_workers和pin_memory等配置的字典
    """
    sys_platform = platform.system()
    
    config = {
        'num_workers': 0,
        'pin_memory': False,
        'persistent_workers': False
    }
    
    # Windows + CUDA: 可以使用多进程和pin_memory
    if sys_platform == 'Windows' and torch.cuda.is_available():
        config['num_workers'] = min(4, os.cpu_count() or 1)
        config['pin_memory'] = True
        config['persistent_workers'] = True if config['num_workers'] > 0 else False
        print(f"[DataLoader配置] Windows + CUDA: num_workers={config['num_workers']}, pin_memory=True")
    
    # Linux + CUDA: 可以使用多进程和pin_memory
    elif sys_platform == 'Linux' and torch.cuda.is_available():
        config['num_workers'] = min(4, os.cpu_count() or 1)
        config['pin_memory'] = True
        config['persistent_workers'] = True if config['num_workers'] > 0 else False
        print(f"[DataLoader配置] Linux + CUDA: num_workers={config['num_workers']}, pin_memory=True")
    
    # macOS + MPS: 不支持pin_memory，使用单进程避免问题
    elif sys_platform == 'Darwin' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        config['num_workers'] = 0  # MPS在多进程下可能有问题
        config['pin_memory'] = False  # MPS不支持pin_memory
        config['persistent_workers'] = False
        print(f"[DataLoader配置] macOS + MPS: num_workers=0, pin_memory=False")
    
    # 其他情况（CPU或其他平台）
    else:
        config['num_workers'] = 0
        config['pin_memory'] = False
        config['persistent_workers'] = False
        print(f"[DataLoader配置] {sys_platform} + CPU: num_workers=0, pin_memory=False")
    
    return config


def load_data_fashion_mnist(batch_size, resize=None, auto_config=True):
    """
    加载Fashion-MNIST数据集
    
    参数:
        batch_size: 批量大小
        resize: 调整图像大小，如果为None则保持原始大小(28x28)
        auto_config: 是否自动配置DataLoader参数（根据平台和设备）
    
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
    
    # 获取DataLoader配置
    if auto_config:
        config = get_dataloader_config()
    else:
        config = {'num_workers': 0, 'pin_memory': False, 'persistent_workers': False}
    
    # 创建数据加载器
    train_iter = DataLoader(
        mnist_train, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        persistent_workers=config['persistent_workers']
    )
    
    test_iter = DataLoader(
        mnist_test, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        persistent_workers=config['persistent_workers']
    )
    
    return train_iter, test_iter

