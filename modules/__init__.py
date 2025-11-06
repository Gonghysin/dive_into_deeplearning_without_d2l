"""
modules 包初始化文件
提供深度学习训练的核心功能
"""

from .trainer import (
    try_gpu,
    get_device_info,
    accuracy,
    evaluate_accuracy,
    train_ch6,
    train_epoch,
    Accumulator,
    Animator,
    detect_available_fonts,
    configure_matplotlib_fonts
)

from .data_loader import (
    load_data_fashion_mnist,
    get_dataloader_config
)

__all__ = [
    # 设备相关
    'try_gpu',
    'get_device_info',
    
    # 训练相关
    'train_ch6',
    'train_epoch',
    'accuracy',
    'evaluate_accuracy',
    
    # 数据加载
    'load_data_fashion_mnist',
    'get_dataloader_config',
    
    # 字体配置
    'detect_available_fonts',
    'configure_matplotlib_fonts',
    
    # 工具类
    'Accumulator',
    'Animator'
]
