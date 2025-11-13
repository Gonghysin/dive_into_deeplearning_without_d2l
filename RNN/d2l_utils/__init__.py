"""
d2l 工具模块
简洁的绘图、数据处理和训练工具，避免复杂的导入依赖
"""

from .plot import plot, Animator, set_figsize, set_axes
from .data import load_array
from .train import train, train_with_animator

__all__ = [
    'plot', 
    'Animator', 
    'set_figsize', 
    'set_axes', 
    'load_array',
    'train',
    'train_with_animator'
]

