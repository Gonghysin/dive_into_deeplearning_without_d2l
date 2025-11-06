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
import os


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
    """
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(8, 6), save_path='training_curves.png'):
        """
        初始化动画绘制器
        
        参数:
            xlabel: x轴标签
            ylabel: y轴标签
            legend: 图例列表
            xlim: x轴范围
            ylim: y轴范围
            xscale: x轴缩放类型
            yscale: y轴缩放类型
            fmts: 线条格式列表
            nrows: 子图行数
            ncols: 子图列数
            figsize: 图像大小
            save_path: 实时保存图片的路径
        """
        # 设置字体（使用系统可用字体，避免字体警告）
        # macOS上优先使用系统字体
        import platform
        if platform.system() == 'Darwin':  # macOS
            plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Arial', 'Helvetica', 'DejaVu Sans']
        else:
            plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 不使用交互模式，提高训练效率
        # plt.ion()  # 已移除，使用非交互式后端
        
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        
        # 设置图片保存路径（使用绝对路径）
        if os.path.isabs(save_path):
            self.save_path = save_path
        else:
            self.save_path = os.path.join(os.getcwd(), save_path)
        self.real_time_save = True  # 是否实时保存图片
        
        if nrows * ncols == 1:
            self.axes = [self.axes]
        # 保存配置参数
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        self.xscale = xscale
        self.yscale = yscale
        self.legend = legend
        self.X, self.Y, self.fmts = None, None, fmts
        
        # 初始化坐标轴（不需要显示窗口）
        self._set_axes(self.axes[0], self.xlabel, self.ylabel, self.xlim, 
                      self.ylim, self.xscale, self.yscale, self.legend)
        
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
        """添加数据点"""
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
        # 清空并重新绘制
        self.axes[0].clear()
        
        # 只有在有数据时才绘制
        has_data = False
        for x_data, y_data, fmt in zip(self.X, self.Y, self.fmts):
            if x_data and y_data:  # 确保有数据
                self.axes[0].plot(x_data, y_data, fmt)
                has_data = True
        
        # 如果没有数据，只显示坐标轴
        if not has_data:
            # 设置一个小的范围以便显示坐标轴
            self.axes[0].set_xlim([0, 1])
            self.axes[0].set_ylim([0, 1])
        
        # 自动调整y轴范围（如果所有数据都已存在）
        if self.X and self.X[0] and has_data:
            all_y_values = []
            for y_list in self.Y:
                if y_list:
                    all_y_values.extend(y_list)
            if all_y_values:
                y_min, y_max = min(all_y_values), max(all_y_values)
                y_range = y_max - y_min
                # 添加一些边距
                y_margin = y_range * 0.1 if y_range > 0 else 0.1
                auto_ylim = [max(0, y_min - y_margin), y_max + y_margin]
                # 使用自动计算的ylim，如果原始ylim为None或者需要自动调整
                use_ylim = auto_ylim if self.ylim is None else self.ylim
            else:
                use_ylim = self.ylim
        else:
            use_ylim = self.ylim
            
        self._set_axes(self.axes[0], self.xlabel, self.ylabel, self.xlim, 
                      use_ylim, self.xscale, self.yscale, self.legend)
        
        # 确保图片有内容，然后保存（不显示窗口，提高训练效率）
        self.fig.tight_layout()  # 自动调整布局
        
        # 实时保存图片到本地 - 只有在有数据时才保存
        if self.real_time_save and has_data:
            try:
                # 确保目录存在
                save_dir = os.path.dirname(self.save_path)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                
                self.fig.savefig(self.save_path, dpi=150, bbox_inches='tight')
                # 验证文件是否真的被创建
                if not os.path.exists(self.save_path):
                    print(f'警告: 图片文件保存失败，路径: {self.save_path}')
            except Exception as e:
                print(f'警告: 保存图片失败: {e}, 路径: {self.save_path}')
    
    def save(self, filename='training_curves.png'):
        """保存图片"""
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f'训练曲线已保存到: {filename}')


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
        animator = Animator(xlabel='Epoch', ylabel='Value', 
                           legend=['训练损失', '训练准确率', '测试准确率'],
                           xlim=[1, num_epochs], ylim=None,  # None表示自动调整
                           figsize=(10, 6), save_path=save_path)
        print(f'训练曲线将实时保存到: {save_path}')
        print('提示: 图片会在后台实时更新，训练完成后可查看上述路径的图片文件')
    
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

