"""
绘图工具模块
提供常用的绘图函数
"""

# 不在这里设置后端，让调用者在导入之前设置
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

# 延迟配置字体，在函数调用时配置，避免导入时崩溃
_font_configured = False

def _configure_fonts():
    """配置中文字体"""
    global _font_configured
    if not _font_configured:
        try:
            plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'Arial Unicode MS', 'SimHei']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            _font_configured = True
        except:
            pass  # 如果配置失败也不影响使用


def plot(x_data, 
         y_data_list, 
         xlabel: str, 
         ylabel: str, 
         legend: Optional[List[str]] = None,
         xlim: Optional[Tuple[float, float]] = None,
         ylim: Optional[Tuple[float, float]] = None,
         figsize: Tuple[float, float] = (6, 3),
         title: Optional[str] = None,
         grid: bool = True,
         save_path: Optional[str] = None):
    """
    绘制折线图
    
    参数:
        x_data: x 轴数据
        y_data_list: y 轴数据列表，可以是单个数组或数组列表
        xlabel: x 轴标签
        ylabel: y 轴标签
        legend: 图例列表（可选）
        xlim: x 轴范围，格式为 (min, max)（可选）
        ylim: y 轴范围，格式为 (min, max)（可选）
        figsize: 图形大小，格式为 (width, height)，默认 (6, 3)
        title: 图标题（可选）
        grid: 是否显示网格，默认 True
        save_path: 保存图片的路径（可选）。如果不提供，则显示图片
    
    示例:
        >>> import torch
        >>> time = torch.arange(0, 100)
        >>> x = torch.sin(0.1 * time)
        >>> plot(time, [x], 'time', 'value', xlim=[0, 100], figsize=(8, 4))
        >>> # 保存图片
        >>> plot(time, [x], 'time', 'value', save_path='output.png')
    """
    # 配置字体（如果还没配置）
    _configure_fonts()
    
    # 确保 y_data_list 是列表格式
    if not isinstance(y_data_list, list):
        y_data_list = [y_data_list]
    
    # 创建图形
    plt.figure(figsize=figsize)
    
    # 绘制每条曲线
    for i, y_data in enumerate(y_data_list):
        label = legend[i] if legend and i < len(legend) else None
        plt.plot(x_data, y_data, label=label)
    
    # 设置标签
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # 设置范围
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    
    # 设置标题
    if title is not None:
        plt.title(title)
    
    # 显示图例（如果提供了）
    if legend is not None:
        plt.legend()
    
    # 显示网格
    if grid:
        plt.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示图形
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 图片已保存到: {save_path}")
        plt.close()
    else:
        plt.show()

