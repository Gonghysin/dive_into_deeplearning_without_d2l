"""
d2l 风格的绘图工具
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def set_figsize(figsize=(3.5, 2.5)):
    """设置图表的尺寸"""
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置坐标轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    if xlim:
        axes.set_xlim(xlim)
    if ylim:
        axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点
    
    参数:
        X: x 轴数据，或当 Y 为 None 时作为 y 轴数据
        Y: y 轴数据（可选），可以是单个数组或数组列表
        xlabel: x 轴标签
        ylabel: y 轴标签
        legend: 图例列表
        xlim: x 轴范围，例如 [0, 100]
        ylim: y 轴范围
        xscale: x 轴刻度类型 ('linear' 或 'log')
        yscale: y 轴刻度类型 ('linear' 或 'log')
        fmts: 线条格式列表
        figsize: 图形大小
        axes: matplotlib axes 对象（可选）
    
    示例:
        >>> import torch
        >>> x = torch.arange(0, 3, 0.01)
        >>> plot(x, [torch.sin(x), torch.cos(x)], 'x', 'f(x)', 
        ...      legend=['sin', 'cos'])
    """
    # 处理输入数据
    if Y is None:
        # 如果没有提供 Y，X 就是 Y 数据，自动生成 X 坐标
        X, Y = [[]] * len(X), X
    
    # 确保 Y 是列表的列表
    if not isinstance(Y, list):
        Y = [Y]
    elif Y and not isinstance(Y[0], (list, tuple, np.ndarray)):
        if not hasattr(Y[0], 'detach'):  # 不是 tensor
            Y = [Y]
    
    # 确保 X 也是列表的列表
    if not isinstance(X, list):
        X = [X] * len(Y)
    elif X and not isinstance(X[0], (list, tuple, np.ndarray)):
        if not hasattr(X[0], 'detach'):  # 不是 tensor
            X = [X] * len(Y)
    
    # 创建图形
    if axes is None:
        fig, axes = plt.subplots(figsize=figsize)
    
    # 清除当前坐标轴
    axes.cla()
    
    # 绘制每条曲线
    for x, y, fmt in zip(X, Y, fmts):
        # 转换为 numpy 数组（处理 PyTorch tensor）
        if hasattr(x, 'detach'):
            x = x.detach().cpu().numpy()
        if hasattr(y, 'detach'):
            y = y.detach().cpu().numpy()
        
        if len(x) if hasattr(x, '__len__') else 0:
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    
    # 设置坐标轴
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    
    # 显示图形
    plt.tight_layout()
    plt.show()
    
    return axes


class Animator:
    """在动画中绘制数据（适用于训练过程可视化）"""
    
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """初始化动画绘图器"""
        # 使用交互模式
        plt.ion()
        
        # 创建图形
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        
        # 配置参数
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
    
    def add(self, x, y):
        """添加数据点到图形"""
        # 将数据转换为列表
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        
        # 初始化数据存储
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        
        # 添加数据
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        
        # 清除并重绘
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        
        # 刷新显示
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
    
    def show(self):
        """显示最终图形"""
        plt.ioff()
        plt.show()


# 配置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

