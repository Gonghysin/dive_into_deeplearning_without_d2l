"""
训练模块
提供模型训练相关的函数
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib
# 使用非交互式后端，提高训练效率，不显示窗口
matplotlib.use('Agg')  # Agg后端，只保存图片，不显示窗口
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import platform
import warnings


def detect_available_fonts(verbose=False):
    """
    检测系统中可用的中英文字体
    
    参数:
        verbose: 是否打印详细信息
    
    返回:
        list: 可用的字体列表（优先级从高到低）
    """
    sys_platform = platform.system()
    
    # 获取系统所有可用字体
    available_fonts = set([f.name for f in fm.fontManager.ttflist])
    
    # 定义各系统的中文字体优先级列表
    font_candidates = {
        'Darwin': [  # macOS
            'Hiragino Sans GB',      # 用户指定的字体
            'PingFang SC',           # 苹方-简体中文
            'PingFang HK',           # 苹方-香港
            'PingFang TC',           # 苹方-繁体中文
            'Heiti SC',              # 黑体-简
            'Heiti TC',              # 黑体-繁
            'STHeiti',               # 华文黑体
            'Arial Unicode MS',      # Arial Unicode（支持中文）
            'Apple LiGothic',        # 苹果俪黑
        ],
        'Windows': [  # Windows
            'Microsoft YaHei',       # 微软雅黑
            'Microsoft YaHei UI',    # 微软雅黑UI
            'SimHei',                # 黑体
            'SimSun',                # 宋体
            'KaiTi',                 # 楷体
            'FangSong',              # 仿宋
            'NSimSun',               # 新宋体
            'YouYuan',               # 幼圆
        ],
        'Linux': [  # Linux
            'Noto Sans CJK SC',      # 思源黑体-简体
            'Noto Sans CJK TC',      # 思源黑体-繁体
            'Noto Serif CJK SC',     # 思源宋体-简体
            'WenQuanYi Micro Hei',   # 文泉驿微米黑
            'WenQuanYi Zen Hei',     # 文泉驿正黑
            'Droid Sans Fallback',   # Droid备用字体
            'AR PL UMing CN',        # 文鼎PL简中明
            'AR PL UKai CN',         # 文鼎PL简中楷
        ]
    }
    
    # 通用备用字体（支持中文的Unicode字体）
    universal_fonts = [
        'DejaVu Sans',
        'Liberation Sans',
        'FreeSans',
        'Arial',
        'Helvetica',
        'sans-serif',  # 系统默认无衬线字体
    ]
    
    # 获取当前系统的字体候选列表
    system_fonts = font_candidates.get(sys_platform, [])
    all_candidates = system_fonts + universal_fonts
    
    # 找出实际可用的字体
    selected_fonts = []
    for font in all_candidates:
        if font in available_fonts or font == 'sans-serif':
            selected_fonts.append(font)
            if verbose and font != 'sans-serif':
                print(f"  ✓ 找到字体: {font}")
    
    # 如果没有找到任何字体，使用系统默认字体
    if not selected_fonts:
        selected_fonts = ['sans-serif']
        if verbose:
            print(f"  ⚠ 未找到推荐字体，使用系统默认字体")
    
    if verbose:
        print(f"[字体配置] {sys_platform} 系统")
        print(f"  已选择字体列表: {selected_fonts[:3]}{'...' if len(selected_fonts) > 3 else ''}")
        print(f"  主要字体: {selected_fonts[0]}")
    
    return selected_fonts


def configure_matplotlib_fonts(verbose=False):
    """
    配置 matplotlib 的字体设置，确保中文显示正常
    
    参数:
        verbose: 是否打印详细信息
    """
    # 获取可用字体列表
    fonts = detect_available_fonts(verbose=verbose)
    
    # 设置字体
    plt.rcParams['font.sans-serif'] = fonts
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    
    # 抑制字体警告
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    if verbose:
        print(f"✓ Matplotlib 字体配置完成\n")
    
    return fonts[0]  # 返回主要字体名称


def get_device_info():
    """
    获取当前系统的设备信息
    
    返回:
        dict: 包含设备类型、平台信息等的字典
    """
    import platform
    
    device_info = {
        'platform': platform.system(),  # 'Darwin' (macOS), 'Windows', 'Linux'
        'platform_version': platform.version(),
        'machine': platform.machine(),  # 'arm64', 'x86_64', etc.
        'has_cuda': torch.cuda.is_available(),
        'cuda_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'has_mps': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
        'mps_built': torch.backends.mps.is_built() if hasattr(torch.backends, 'mps') else False,
    }
    
    # 添加 CUDA 设备详情
    if device_info['has_cuda']:
        device_info['cuda_devices'] = []
        for i in range(device_info['cuda_count']):
            device_info['cuda_devices'].append({
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'capability': torch.cuda.get_device_capability(i),
                'total_memory': f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB"
            })
    
    return device_info


def try_gpu(i=0, verbose=True):
    """
    尝试获取GPU设备，如果不可用则返回CPU
    自动检测并选择最佳设备：
    - macOS (Apple Silicon): 使用 MPS (Metal Performance Shaders)
    - Windows/Linux (NVIDIA GPU): 使用 CUDA
    - 其他情况: 使用 CPU
    
    参数:
        i: GPU设备索引，默认为0（对CUDA有效，MPS总是使用单设备）
        verbose: 是否打印详细设备信息
    
    返回:
        torch.device对象
    """
    device_info = get_device_info()
    
    # 优先使用Mac的MPS（Metal Performance Shaders）
    if device_info['has_mps']:
        device = torch.device('mps')
        if verbose:
            print(f"✓ 检测到 Apple Silicon GPU (MPS)")
            print(f"  平台: {device_info['platform']} ({device_info['machine']})")
            print(f"  使用设备: {device}")
    # 其次尝试CUDA（NVIDIA GPU）
    elif device_info['has_cuda'] and device_info['cuda_count'] >= i + 1:
        device = torch.device(f'cuda:{i}')
        if verbose:
            print(f"✓ 检测到 NVIDIA GPU (CUDA)")
            print(f"  平台: {device_info['platform']}")
            cuda_device = device_info['cuda_devices'][i]
            print(f"  GPU {i}: {cuda_device['name']}")
            print(f"  显存: {cuda_device['total_memory']}")
            print(f"  计算能力: {cuda_device['capability']}")
            print(f"  使用设备: {device}")
    # 最后回退到CPU
    else:
        device = torch.device('cpu')
        if verbose:
            print(f"✗ 未检测到可用 GPU，使用 CPU")
            print(f"  平台: {device_info['platform']} ({device_info['machine']})")
            if device_info['platform'] == 'Darwin' and not device_info['has_mps']:
                print(f"  提示: 如果您使用的是 Apple Silicon Mac，请确保 PyTorch 版本支持 MPS")
            elif not device_info['has_cuda']:
                print(f"  提示: 未检测到 CUDA，如需使用 NVIDIA GPU 请安装 CUDA 版本的 PyTorch")
            print(f"  使用设备: {device}")
    
    return device


def accuracy(y_hat, y):
    """
    计算预测准确率
    
    参数:
        y_hat: 模型预测结果，形状为 (batch_size, num_classes)
        y: 真实标签，形状为 (batch_size,)
    
    返回:
        准确率（标量）
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter, device, desc=None):
    """
    评估模型在数据集上的准确率
    
    参数:
        net: 神经网络模型
        data_iter: 数据迭代器
        device: 计算设备
        desc: 进度条描述（可选）
    
    返回:
        准确率（标量）
    """
    net.eval()  # 设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    
    with torch.no_grad():
        # 使用tqdm显示进度条，dynamic_ncols=True自动适应终端宽度，避免换行
        pbar = tqdm(data_iter, desc=desc, leave=False, dynamic_ncols=True, mininterval=0.1)
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
            # 更新进度条信息
            current_acc = metric[0] / metric[1]
            pbar.set_postfix({'acc': f'{current_acc:.4f}'})
    
    return metric[0] / metric[1]


class Accumulator:
    """
    累加器类，用于累加多个指标
    """
    def __init__(self, n):
        self.data = [0.0] * n
    
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class Animator:
    """
    实时绘制训练曲线的动画类
    支持多子图，可以将不同类型的指标分开显示
    """
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(8, 6), save_path='training_curves.png',
                 subplot_config=None):
        """
        初始化动画绘制器
        
        参数:
            xlabel: x轴标签（单个标签或标签列表，对应每个子图）
            ylabel: y轴标签（单个标签或标签列表，对应每个子图）
            legend: 图例列表（单个列表或嵌套列表，对应每个子图）
            xlim: x轴范围
            ylim: y轴范围（单个范围或范围列表，对应每个子图）
            xscale: x轴缩放类型
            yscale: y轴缩放类型
            fmts: 线条格式列表
            nrows: 子图行数
            ncols: 子图列数
            figsize: 图像大小
            save_path: 实时保存图片的路径
            subplot_config: 子图配置字典列表，每个字典包含该子图的配置
                          [{'data_indices': [0], 'ylabel': 'Loss', 'legend': ['Train Loss']}, ...]
        """
        # 自动配置字体，支持 Linux/Windows/Mac 等各种系统
        self.primary_font = configure_matplotlib_fonts(verbose=False)
        
        # 不使用交互模式，提高训练效率
        # plt.ion()  # 已移除，使用非交互式后端
        
        if legend is None:
            legend = []
        
        self.nrows = nrows
        self.ncols = ncols
        self.num_subplots = nrows * ncols
        
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        
        # 设置图片保存路径（使用绝对路径）
        if os.path.isabs(save_path):
            self.save_path = save_path
        else:
            self.save_path = os.path.join(os.getcwd(), save_path)
        self.real_time_save = True  # 是否实时保存图片
        
        # 确保 axes 是列表
        if nrows * ncols == 1:
            self.axes = [self.axes]
        else:
            self.axes = self.axes.flatten()
        
        # 保存配置参数
        self.xlabel = xlabel if isinstance(xlabel, list) else [xlabel] * self.num_subplots
        self.ylabel = ylabel if isinstance(ylabel, list) else [ylabel] * self.num_subplots
        self.legend = legend if (isinstance(legend, list) and legend and isinstance(legend[0], list)) else [legend]
        self.xlim = xlim
        self.ylim = ylim if isinstance(ylim, list) else [ylim] * self.num_subplots
        self.xscale = xscale
        self.yscale = yscale
        self.fmts = fmts
        self.X, self.Y = None, None
        
        # 子图配置：指定哪些数据系列绘制在哪个子图上
        self.subplot_config = subplot_config
        
        # 初始化所有子图的坐标轴
        for i, ax in enumerate(self.axes):
            leg = self.legend[i] if i < len(self.legend) else []
            self._set_axes(ax, self.xlabel[i], self.ylabel[i], self.xlim, 
                          self.ylim[i], self.xscale, self.yscale, leg)
        
    def _set_axes(self, axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
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
    
    def add(self, x, y):
        """
        添加数据点
        
        参数:
            x: x轴数据（标量或列表）
            y: y轴数据（列表，包含所有数据系列的值）
        """
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        
        # 如果有子图配置，按配置绘制到不同子图
        if self.subplot_config:
            has_data = False
            for subplot_idx, config in enumerate(self.subplot_config):
                if subplot_idx >= len(self.axes):
                    break
                    
                ax = self.axes[subplot_idx]
                ax.clear()
                
                data_indices = config.get('data_indices', [])
                subplot_has_data = False
                
                # 绘制该子图对应的数据系列
                for i, data_idx in enumerate(data_indices):
                    if data_idx < len(self.X) and self.X[data_idx] and self.Y[data_idx]:
                        fmt = self.fmts[i % len(self.fmts)]
                        ax.plot(self.X[data_idx], self.Y[data_idx], fmt, linewidth=2)
                        subplot_has_data = True
                        has_data = True
                
                # 设置该子图的属性
                if subplot_has_data:
                    # 自动调整y轴范围
                    subplot_y_values = []
                    for data_idx in data_indices:
                        if data_idx < len(self.Y) and self.Y[data_idx]:
                            subplot_y_values.extend(self.Y[data_idx])
                    
                    # 过滤掉 NaN 和 Inf 值
                    import math
                    valid_y_values = [y for y in subplot_y_values if not (math.isnan(y) or math.isinf(y))]
                    
                    if valid_y_values:
                        y_min, y_max = min(valid_y_values), max(valid_y_values)
                        y_range = y_max - y_min
                        y_margin = y_range * 0.1 if y_range > 0 else 0.1
                        auto_ylim = [max(0, y_min - y_margin), y_max + y_margin]
                        use_ylim = auto_ylim if self.ylim[subplot_idx] is None else self.ylim[subplot_idx]
                    else:
                        # 如果所有值都是 NaN/Inf，使用默认范围
                        use_ylim = [0, 1] if self.ylim[subplot_idx] is None else self.ylim[subplot_idx]
                        print(f'警告: 子图 {subplot_idx} 的所有 y 值都是 NaN 或 Inf，训练可能不稳定！')
                else:
                    use_ylim = self.ylim[subplot_idx]
                
                # 设置坐标轴
                subplot_legend = config.get('legend', [])
                self._set_axes(ax, self.xlabel[subplot_idx], config.get('ylabel', self.ylabel[subplot_idx]), 
                              self.xlim, use_ylim, self.xscale, self.yscale, subplot_legend)
        else:
            # 原有逻辑：所有数据绘制在第一个子图
            self.axes[0].clear()
            has_data = False
            for x_data, y_data, fmt in zip(self.X, self.Y, self.fmts):
                if x_data and y_data:
                    self.axes[0].plot(x_data, y_data, fmt, linewidth=2)
                    has_data = True
            
            if not has_data:
                self.axes[0].set_xlim([0, 1])
                self.axes[0].set_ylim([0, 1])
            
            # 自动调整y轴范围
            if self.X and self.X[0] and has_data:
                all_y_values = []
                for y_list in self.Y:
                    if y_list:
                        all_y_values.extend(y_list)
                
                # 过滤掉 NaN 和 Inf 值
                import math
                valid_y_values = [y for y in all_y_values if not (math.isnan(y) or math.isinf(y))]
                
                if valid_y_values:
                    y_min, y_max = min(valid_y_values), max(valid_y_values)
                    y_range = y_max - y_min
                    y_margin = y_range * 0.1 if y_range > 0 else 0.1
                    auto_ylim = [max(0, y_min - y_margin), y_max + y_margin]
                    use_ylim = auto_ylim if self.ylim[0] is None else self.ylim[0]
                else:
                    # 如果所有值都是 NaN/Inf，使用默认范围
                    use_ylim = [0, 1] if self.ylim[0] is None else self.ylim[0]
                    print(f'警告: 所有 y 值都是 NaN 或 Inf，训练可能不稳定！')
            else:
                use_ylim = self.ylim[0]
                
            self._set_axes(self.axes[0], self.xlabel[0], self.ylabel[0], self.xlim, 
                          use_ylim, self.xscale, self.yscale, self.legend[0] if self.legend else [])
        
        # 确保图片有内容，然后保存
        self.fig.tight_layout()
        
        # 实时保存图片到本地
        if self.real_time_save and has_data:
            try:
                save_dir = os.path.dirname(self.save_path)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                
                self.fig.savefig(self.save_path, dpi=150, bbox_inches='tight')
                if not os.path.exists(self.save_path):
                    print(f'Warning: Failed to save plot to: {self.save_path}')
            except Exception as e:
                print(f'Warning: Failed to save plot: {e}, path: {self.save_path}')
    
    def save(self, filename='training_curves.png'):
        """保存图片"""
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f'Training curve saved to: {filename}')


def train_epoch(net, train_iter, loss, optimizer, device, desc=None):
    """
    训练一个epoch
    
    参数:
        net: 神经网络模型
        train_iter: 训练数据迭代器
        loss: 损失函数
        optimizer: 优化器
        device: 计算设备
        desc: 进度条描述（可选）
    
    返回:
        训练损失和准确率
    """
    net.train()  # 设置为训练模式
    metric = Accumulator(3)  # 训练损失总和、训练准确度总和、样本数
    
    # 使用tqdm显示进度条，dynamic_ncols=True自动适应终端宽度，避免换行
    pbar = tqdm(train_iter, desc=desc, leave=False, dynamic_ncols=True, mininterval=0.1)
    for X, y in pbar:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = net(X)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()
        
        # 更新指标
        metric.add(float(l) * len(y), accuracy(y_hat, y), y.numel())
        
        # 更新进度条信息
        current_loss = metric[0] / metric[2]
        current_acc = metric[1] / metric[2]
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.4f}'
        })
    
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device=None, show_plot=True):
    """
    训练模型（第6章版本）
    
    参数:
        net: 神经网络模型
        train_iter: 训练数据迭代器
        test_iter: 测试数据迭代器
        num_epochs: 训练轮数
        lr: 学习率
        device: 计算设备（如果为None，则自动检测最佳设备）
        show_plot: 是否显示训练曲线图（默认True）
    
    返回:
        训练历史记录字典，包含train_loss, train_acc, test_acc列表
    """
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    
    net.apply(init_weights)
    
    # 如果没有指定设备，则自动检测
    if device is None:
        print("=" * 60)
        print("自动检测计算设备...")
        print("=" * 60)
        device = try_gpu(verbose=True)
        print("=" * 60)
    else:
        print(f'使用指定设备: {device}')
    
    # 配置绘图字体（在初始化 Animator 之前）
    if show_plot:
        print("\n" + "=" * 60)
        print("配置绘图字体...")
        print("=" * 60)
        configure_matplotlib_fonts(verbose=True)
        print("=" * 60)
    
    print(f'\n开始训练模型...')
    
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    
    # 初始化训练历史记录
    train_losses = []
    train_accs = []
    test_accs = []
    
    # 初始化动画绘制器（如果启用）
    if show_plot:
        # 获取保存路径（保存在项目根目录 deep_to_dl 下）
        # 从 trainer.py 的位置向上找到项目根目录
        current_file = os.path.abspath(__file__)  # trainer.py 的绝对路径
        modules_dir = os.path.dirname(current_file)  # modules 目录
        project_root = os.path.dirname(modules_dir)  # deep_to_dl 目录
        save_path = os.path.join(project_root, 'training_curves.png')
        
        # 使用两个子图：上方显示损失，下方显示准确率
        # 数据顺序：[0: train_loss, 1: train_acc, 2: test_acc]
        subplot_config = [
            {
                'data_indices': [0],  # 训练损失
                'ylabel': 'Loss',
                'legend': ['Train Loss']
            },
            {
                'data_indices': [1, 2],  # 训练和测试准确率
                'ylabel': 'Accuracy',
                'legend': ['Train Acc', 'Test Acc']
            }
        ]
        
        animator = Animator(
            xlabel=['Epoch', 'Epoch'],
            ylabel=['Loss', 'Accuracy'],
            xlim=[1, num_epochs],
            ylim=[None, [0, 1]],  # 损失自动调整，准确率固定0-1
            nrows=2, ncols=1,
            figsize=(10, 8),
            save_path=save_path,
            subplot_config=subplot_config
        )
        print(f'Training curve will be saved to: {save_path}')
        print('Note: The plot will be updated in real-time during training')
    
    # 训练循环
    epoch_pbar = tqdm(range(num_epochs), desc='总体进度', dynamic_ncols=True)
    for epoch in epoch_pbar:
        # 训练一个epoch
        train_metrics = train_epoch(net, train_iter, loss, optimizer, device, 
                                    desc=f'Epoch {epoch+1}/{num_epochs} [训练]')
        train_loss, train_acc = train_metrics
        
        # 评估测试集
        test_acc = evaluate_accuracy(net, test_iter, device, 
                                    desc=f'Epoch {epoch+1}/{num_epochs} [测试]')
        
        # 记录训练历史
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        # 更新训练曲线
        if show_plot:
            animator.add(epoch + 1, (train_loss, train_acc, test_acc))
            # 提示用户图片已更新
            if epoch == 0 or (epoch + 1) % max(1, num_epochs // 5) == 0:
                if os.path.exists(animator.save_path):
                    print(f'\n[Epoch {epoch+1}] 训练曲线已更新并保存: {animator.save_path}')
                else:
                    print(f'\n[Epoch {epoch+1}] 警告: 图片文件未找到: {animator.save_path}')
        
        # 更新总体进度条
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'train_acc': f'{train_acc:.4f}',
            'test_acc': f'{test_acc:.4f}'
        })
    
    # 训练完成后的验证
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
    
    print(f'\n训练完成！最终测试准确率: {test_acc:.4f}')
    
    # 保存最终训练曲线图（已经在每个epoch保存过了，这里确保最终版本已保存）
    if show_plot:
        # 最终保存一次，确保图片是最新的
        if os.path.exists(animator.save_path):
            print(f'\n训练曲线已保存到: {animator.save_path}')
        else:
            animator.save(animator.save_path)
    
    # 返回训练历史
    return {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'test_acc': test_accs
    }

