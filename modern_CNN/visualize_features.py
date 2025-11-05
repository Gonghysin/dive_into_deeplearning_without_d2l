"""
可视化程序：展示数据集样本和模型特征图
"""
import sys
import os
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from modules.trainer import try_gpu
import platform

# 设置matplotlib后端（不使用GUI）
import matplotlib
matplotlib.use('Agg')

# 设置字体
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Arial', 'Helvetica', 'DejaVu Sans']
else:
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Fashion-MNIST类别名称
FASHION_MNIST_CLASSES = [
    'T-shirt/top',      # 0
    'Trouser',          # 1
    'Pullover',         # 2
    'Dress',            # 3
    'Coat',             # 4
    'Sandal',           # 5
    'Shirt',            # 6
    'Sneaker',          # 7
    'Bag',              # 8
    'Ankle boot'        # 9
]


def get_model():
    """获取AlexNet模型定义"""
    net = nn.Sequential(
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(),
        nn.Linear(6400, 4096), nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096), nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 10)
    )
    return net


def get_sample_images_per_class(dataset, num_classes=10):
    """
    获取每个类别的一个样本图片
    
    参数:
        dataset: Fashion-MNIST数据集
        num_classes: 类别数量
    
    返回:
        samples: 字典，{class_id: (image_tensor, label)}
    """
    samples = {}
    for i in range(len(dataset)):
        image, label = dataset[i]
        if label not in samples:
            samples[label] = (image, label)
        if len(samples) == num_classes:
            break
    return samples


def register_hooks(net):
    """
    注册forward hook来捕获每层的输出
    
    返回:
        activations: 字典，存储每层的输出
        hooks: hook列表，用于清理
    """
    activations = {}
    hooks = []
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    layer_idx = 0
    for name, module in net.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ReLU, nn.MaxPool2d)):
            hook = module.register_forward_hook(get_activation(f'layer_{layer_idx}_{module.__class__.__name__}'))
            hooks.append(hook)
            layer_idx += 1
    
    return activations, hooks


def visualize_class_samples(samples, save_path='class_samples.png'):
    """
    可视化每个类别的样本图片
    
    参数:
        samples: 字典，{class_id: (image_tensor, label)}
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for class_id in range(10):
        if class_id in samples:
            image, label = samples[class_id]
            # 转换为numpy数组
            img = image.squeeze().numpy()
            
            axes[class_id].imshow(img, cmap='gray')
            axes[class_id].set_title(f'{class_id}: {FASHION_MNIST_CLASSES[label]}', fontsize=10)
            axes[class_id].axis('off')
    
    plt.suptitle('Fashion-MNIST 数据集类别样本', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'类别样本图片已保存到: {save_path}')
    plt.close()


def visualize_feature_maps(image, activations, class_name, save_dir='feature_maps'):
    """
    可视化单张图片经过各层后的特征图
    
    参数:
        image: 输入图片张量 (1, 1, H, W)
        activations: 各层的激活值字典
        class_name: 类别名称
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 为每个卷积层创建特征图可视化
    conv_layers = []
    for name, activation in activations.items():
        if 'Conv2d' in name:
            conv_layers.append((name, activation))
    
    if not conv_layers:
        print(f'未找到卷积层特征图')
        return
    
    # 为每个卷积层创建一个图
    for layer_name, features in conv_layers:
        # features shape: (batch, channels, height, width)
        features = features[0]  # 取第一个batch
        num_channels = features.shape[0]
        
        # 选择前16个通道进行可视化（如果通道数超过16）
        num_vis = min(16, num_channels)
        channels_to_vis = list(range(num_vis))
        
        # 计算网格大小
        cols = 4
        rows = (num_vis + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, channel_idx in enumerate(channels_to_vis):
            feature_map = features[channel_idx].cpu().numpy()
            axes[idx].imshow(feature_map, cmap='viridis')
            axes[idx].set_title(f'Channel {channel_idx}', fontsize=8)
            axes[idx].axis('off')
        
        # 隐藏多余的子图
        for idx in range(len(channels_to_vis), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'{class_name} - {layer_name}\n特征图 (前{num_vis}个通道)', 
                    fontsize=10, fontweight='bold')
        plt.tight_layout()
        
        # 保存图片
        safe_name = layer_name.replace('/', '_')
        save_path = os.path.join(save_dir, f'{class_name}_{safe_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f'{class_name} 的特征图已保存到: {save_dir}')


def visualize_all_layers_summary(image, activations, class_name, save_path='layer_summary.png'):
    """
    创建所有层的摘要可视化（每层选择一个代表性特征图）
    
    参数:
        image: 输入图片
        activations: 各层的激活值
        class_name: 类别名称
        save_path: 保存路径
    """
    # 收集所有层的输出
    layers = []
    for name, activation in sorted(activations.items()):
        if 'Conv2d' in name or 'MaxPool2d' in name:
            layers.append((name, activation))
    
    if not layers:
        print('未找到可可视化的层')
        return
    
    # 创建一个大图，包含原始图片和每层的特征图
    num_layers = len(layers) + 1  # +1 for original image
    cols = 4
    rows = (num_layers + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # 显示原始图片
    img = image.squeeze().cpu().numpy()
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('原始图片', fontsize=10, fontweight='bold')
    axes[0].axis('off')
    
    # 显示每层的特征图（选择第0个通道）
    for idx, (layer_name, features) in enumerate(layers, start=1):
        if idx >= len(axes):
            break
        
        feat = features[0]  # 取第一个batch
        
        if len(feat.shape) == 3:  # (channels, height, width)
            # 选择第一个通道
            feature_map = feat[0].cpu().numpy()
            axes[idx].imshow(feature_map, cmap='viridis')
            axes[idx].set_title(f'{layer_name}\nShape: {feat.shape}', fontsize=8)
            axes[idx].axis('off')
    
    # 隐藏多余的子图
    for idx in range(len(layers) + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'{class_name} - 各层特征图摘要', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'层摘要图已保存到: {save_path}')
    plt.close()


def main():
    """主函数"""
    print("=" * 60)
    print("Fashion-MNIST 数据集和模型特征图可视化")
    print("=" * 60)
    
    # 设置设备
    device = try_gpu()
    print(f'使用设备: {device}')
    
    # 加载数据集
    print('\n1. 加载数据集...')
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])
    
    dataset = datasets.FashionMNIST(
        root=data_dir, train=False, transform=transform, download=True
    )
    
    # 获取每个类别的样本
    print('2. 获取每个类别的样本图片...')
    samples = get_sample_images_per_class(dataset)
    
    # 可视化类别样本
    print('3. 可视化类别样本...')
    visualize_class_samples(samples, save_path='class_samples.png')
    
    # 加载模型（使用随机初始化的权重，如果是训练后的模型需要加载权重）
    print('\n4. 加载模型...')
    net = get_model()
    net.to(device)
    net.eval()
    
    # 注册hook来捕获特征图
    print('5. 注册特征图捕获hook...')
    activations, hooks = register_hooks(net)
    
    # 为每个类别可视化特征图
    print('\n6. 可视化各层特征图...')
    print('   这可能需要一些时间...')
    
    os.makedirs('feature_maps', exist_ok=True)
    
    for class_id in range(10):
        if class_id not in samples:
            continue
        
        image, label = samples[class_id]
        class_name = FASHION_MNIST_CLASSES[label]
        
        print(f'   处理类别 {class_id}: {class_name}...')
        
        # 清空之前的激活值
        activations.clear()
        
        # 准备输入
        input_tensor = image.unsqueeze(0).to(device)  # (1, 1, H, W)
        
        # 前向传播
        with torch.no_grad():
            _ = net(input_tensor)
        
        # 可视化详细特征图
        visualize_feature_maps(input_tensor, activations, class_name, save_dir='feature_maps')
        
        # 可视化层摘要
        summary_path = os.path.join('feature_maps', f'{class_name}_layer_summary.png')
        visualize_all_layers_summary(input_tensor, activations, class_name, save_path=summary_path)
    
    # 清理hooks
    for hook in hooks:
        hook.remove()
    
    print('\n' + "=" * 60)
    print("可视化完成！")
    print("=" * 60)
    print("\n生成的文件：")
    print("1. class_samples.png - 每个类别的样本图片")
    print("2. feature_maps/ - 各层特征图目录")
    print("   - 每个类别都有详细的卷积层特征图")
    print("   - 每个类别都有层摘要图")


if __name__ == '__main__':
    main()



