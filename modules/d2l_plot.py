"""
d2l 风格的绘图工具模块
独立模块，不依赖其他 modules 子模块，避免导入冲突
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def use_svg_display():
    """使用 svg 格式显示图片（用于 jupyter，在 py 文件中无效）"""
    try:
        from matplotlib_inline import backend_inline
        backend_inline.set_matplotlib_formats('svg')
    except:
        pass  # 在非 jupyter 环境中忽略


def set_figsize(figsize=(3.5, 2.5)):
    """设置图表的尺寸
    
    参数:
        figsize: 图形大小 (width, height)
    """
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置坐标轴
    
    参数:
        axes: matplotlib axes 对象
        xlabel: x 轴标签
        ylabel: y 轴标签
        xlim: x 轴范围
        ylim: y 轴范围
        xscale: x 轴刻度类型 ('linear' 或 'log')
        yscale: y 轴刻度类型 ('linear' 或 'log')
        legend: 图例列表
    """
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
        Y: y 轴数据（可选）
        xlabel: x 轴标签
        ylabel: y 轴标签
        legend: 图例列表
        xlim: x 轴范围
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
    # 如果 Y 没有提供，假设 X 是 Y，生成对应的 X 坐标
    if Y is None:
        X, Y = [[]] * len(X), X
    elif not isinstance(Y[0], (list, tuple, np.ndarray)):
        # 如果 Y 不是列表的列表，转换为列表的列表
        if hasattr(Y[0], 'detach'):  # PyTorch tensor
            Y = [Y]
        else:
            Y = [Y]
    
    # 确保 X 也是列表的列表
    if not isinstance(X[0], (list, tuple, np.ndarray)):
        if hasattr(X[0], 'detach'):  # PyTorch tensor
            X = [X] * len(Y)
        else:
            X = [X] * len(Y)
    
    # 创建图形
    if axes is None:
        fig, axes = plt.subplots(figsize=figsize)
    
    # 清除当前坐标轴（用于动画）
    axes.cla()
    
    # 绘制每条曲线
    for x, y, fmt in zip(X, Y, fmts):
        # 转换为 numpy 数组（处理 PyTorch tensor）
        if hasattr(x, 'detach'):
            x = x.detach().cpu().numpy()
        if hasattr(y, 'detach'):
            y = y.detach().cpu().numpy()
        
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    
    # 设置坐标轴
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    
    return axes


class Animator:
    """在动画中绘制数据（适用于训练过程可视化）"""
    
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """初始化动画绘图器
        
        参数:
            xlabel: x 轴标签
            ylabel: y 轴标签
            legend: 图例列表
            xlim: x 轴范围
            ylim: y 轴范围
            xscale: x 轴刻度类型
            yscale: y 轴刻度类型
            fmts: 线条格式列表
            nrows: 子图行数
            ncols: 子图列数
            figsize: 图形大小
        """
        # 在 py 文件中使用交互模式
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
        """添加数据点到图形
        
        参数:
            x: x 坐标
            y: y 坐标（可以是单个值或列表）
        """
        # 将数据转换为浮点数（处理 tensor）
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


# 配置 matplotlib 中文字体
def configure_chinese_fonts():
    """配置中文字体支持"""
    try:
        plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass


# 初始化时配置字体
configure_chinese_fonts()

