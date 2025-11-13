"""
d2l 工具模块
简洁的绘图、数据处理、训练和文本处理工具，避免复杂的导入依赖
"""

from .plot import plot, Animator, set_figsize, set_axes
from .data import load_array
from .train import train, train_with_animator
from .download import DATA_HUB, DATA_URL, download, register_dataset
from .text import read_time_machine, tokenize, count_corpus, Vocab, load_corpus_time_machine, seq_data_iter_random, seq_data_iter_sequential, SeqDataLoader, load_data_time_machine

__all__ = [
    # 绘图工具
    'plot', 
    'Animator', 
    'set_figsize', 
    'set_axes',
    
    # 数据加载
    'load_array',
    
    # 训练工具
    'train',
    'train_with_animator',
    
    # 下载工具
    'DATA_HUB',
    'DATA_URL',
    'download',
    'register_dataset',
    
    # 文本处理
    'read_time_machine',
    'tokenize',
    'count_corpus',
    'Vocab',
    'load_corpus_time_machine',
    'seq_data_iter_random',
    'seq_data_iter_sequential',
    'SeqDataLoader',
    'load_data_time_machine',
]

