"""
modules 包初始化文件
提供深度学习训练的核心功能

特性：
- 支持通过 @save 装饰器动态注册函数/类到模块中
- 自动管理 __all__ 列表
"""

import sys
from functools import wraps

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

# 动态注册表：用于存储通过 @save 装饰器注册的对象
_registry = {}

def save(obj=None, *, name=None, category="custom"):
    """
    装饰器：将函数/类保存到 modules 模块中，使其可以被导入使用
    
    使用方式：
        @save
        def my_function():
            pass
        
        @save(name="custom_name", category="models")
        class MyModel:
            pass
    
    参数：
        obj: 被装饰的对象（函数或类）
        name: 自定义名称（默认使用对象的 __name__）
        category: 分类标签（用于组织和查找，默认为 "custom"）
    
    示例：
        # 在 resnet.py 中
        from modules import save
        
        @save
        def residual_block(in_channels, out_channels):
            return nn.Sequential(...)
        
        # 在其他文件中
        from modules import residual_block  # 直接导入使用！
    """
    def decorator(obj):
        # 获取对象名称
        obj_name = name if name is not None else obj.__name__
        
        # 注册到注册表
        _registry[obj_name] = {
            'object': obj,
            'category': category,
            'type': 'function' if callable(obj) and not isinstance(obj, type) else 'class'
        }
        
        # 添加到当前模块的命名空间
        current_module = sys.modules[__name__]
        setattr(current_module, obj_name, obj)
        
        # 动态更新 __all__
        if obj_name not in __all__:
            __all__.append(obj_name)
        
        # 添加元信息
        if not hasattr(obj, '__registry_info__'):
            obj.__registry_info__ = {
                'registered_as': obj_name,
                'category': category,
                'module': 'modules'
            }
        
        print(f"✓ 已注册 {_registry[obj_name]['type']}: {obj_name} (category: {category})")
        
        return obj
    
    # 支持 @save 和 @save(...) 两种用法
    if obj is None:
        # @save(name="...", category="...")
        return decorator
    else:
        # @save
        return decorator(obj)


def list_registered(category=None, show_details=False):
    """
    列出所有通过 @save 注册的对象
    
    参数：
        category: 只显示指定分类的对象（None 表示显示全部）
        show_details: 是否显示详细信息
    
    返回：
        注册对象的列表或详细信息字典
    """
    if category:
        filtered = {k: v for k, v in _registry.items() if v['category'] == category}
    else:
        filtered = _registry
    
    if show_details:
        return filtered
    else:
        return list(filtered.keys())


def get_registry():
    """
    获取完整的注册表
    
    返回：
        包含所有注册对象的字典
    """
    return _registry.copy()


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
    'Animator',
    
    # 装饰器和工具函数
    'save',
    'list_registered',
    'get_registry',
]
