"""
设备检测和管理模块
"""

import torch


def try_gpu(i=0):
    """如果存在，返回 gpu(i)，否则返回 cpu()
    
    参数:
        i: GPU 设备索引，默认为 0
    
    返回:
        torch.device: 可用的设备对象
    
    示例:
        >>> device = try_gpu()
        >>> print(device)
        cuda:0  # 如果有 GPU
        # 或
        cpu  # 如果没有 GPU
    """
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    """返回所有可用的 GPU，如果没有 GPU，则返回 [cpu()]
    
    返回:
        list: 设备列表
    
    示例:
        >>> devices = try_all_gpus()
        >>> print(devices)
        [device(type='cuda', index=0), device(type='cuda', index=1)]
        # 或
        [device(type='cpu')]
    """
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


def get_device_info():
    """获取设备信息
    
    返回:
        dict: 包含设备信息的字典
    
    示例:
        >>> info = get_device_info()
        >>> print(info)
        {'device': 'cuda:0', 'num_gpus': 1, 'gpu_name': 'NVIDIA GeForce RTX 3080'}
    """
    info = {
        'num_gpus': torch.cuda.device_count(),
        'device': str(try_gpu()),
    }
    
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    
    return info

