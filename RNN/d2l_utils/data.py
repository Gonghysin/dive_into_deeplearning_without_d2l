"""
数据处理工具
"""

import torch
from torch.utils.data import TensorDataset, DataLoader


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个 PyTorch 数据迭代器
    
    参数:
        data_arrays: 包含特征和标签的元组，例如 (features, labels)
        batch_size: 批量大小
        is_train: 是否为训练模式（训练时打乱数据）
    
    返回:
        DataLoader: PyTorch 数据加载器
    
    示例:
        >>> features = torch.randn(1000, 10)
        >>> labels = torch.randn(1000, 1)
        >>> train_iter = load_array((features, labels), batch_size=32, is_train=True)
        >>> for X, y in train_iter:
        ...     print(X.shape, y.shape)
        ...     break
    """
    # 将所有数据转换为 tensor（如果还不是的话）
    tensors = []
    for data in data_arrays:
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        tensors.append(data)
    
    # 创建 TensorDataset
    dataset = TensorDataset(*tensors)
    
    # 创建 DataLoader
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=is_train,
        num_workers=0  # 设置为 0 避免多进程问题
    )

