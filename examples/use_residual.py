"""
示例: 在新程序中使用通过 @save 注册的 Residual 类
"""
import torch
import torch.nn as nn
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 重要：首次使用时需要导入 resnet 来触发注册
print("导入 resnet.py 以注册 Residual 类...")
import modern_CNN.resnet

# 现在可以直接从 modules 导入使用！
print("从 modules 导入 Residual...")
from modules import Residual

print("\n" + "=" * 60)
print("示例 1: 创建基本的残差块")
print("=" * 60)

# 创建一个标准的残差块（输入输出通道数相同）
block1 = Residual(input_channels=64, num_channels=64)
print(f"\n残差块 1: {block1}")

X1 = torch.randn(8, 64, 56, 56)  # batch_size=8, channels=64, H=56, W=56
Y1 = block1(X1)

print(f"\n输入形状: {X1.shape}")
print(f"输出形状: {Y1.shape}")
print("✓ 维度保持不变")

print("\n" + "=" * 60)
print("示例 2: 创建改变维度的残差块")
print("=" * 60)

# 创建一个改变通道数和尺寸的残差块
block2 = Residual(input_channels=64, num_channels=128, 
                  use_1x1conv=True, strides=2)
print(f"\n残差块 2: {block2}")

X2 = torch.randn(8, 64, 56, 56)
Y2 = block2(X2)

print(f"\n输入形状: {X2.shape}")
print(f"输出形状: {Y2.shape}")
print("✓ 通道数 64→128, 尺寸 56→28 (stride=2)")

print("\n" + "=" * 60)
print("示例 3: 构建完整的 ResNet 风格网络")
print("=" * 60)

class MyResNet(nn.Module):
    """使用注册的 Residual 块构建自定义 ResNet"""
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 使用注册的 Residual 块！
        self.layer1 = nn.Sequential(
            Residual(64, 64),
            Residual(64, 64)
        )
        
        self.layer2 = nn.Sequential(
            Residual(64, 128, use_1x1conv=True, strides=2),
            Residual(128, 128)
        )
        
        self.layer3 = nn.Sequential(
            Residual(128, 256, use_1x1conv=True, strides=2),
            Residual(256, 256)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 创建模型
model = MyResNet(num_classes=10)
print(f"\n模型结构:\n{model}")

# 测试
X = torch.randn(2, 1, 224, 224)
output = model(X)
print(f"\n测试前向传播:")
print(f"  输入形状: {X.shape}")
print(f"  输出形状: {output.shape}")

# 统计参数
total_params = sum(p.numel() for p in model.parameters())
print(f"  总参数量: {total_params:,}")

print("\n" + "=" * 60)
print("✓ 完成！成功使用 @save 注册的 Residual 类构建模型")
print("=" * 60)

print("\n使用建议:")
print("  1. 在项目启动时导入一次 resnet.py 来注册所有组件")
print("  2. 之后就可以像使用普通模块一样使用 Residual")
print("  3. 你可以在多个文件中重复使用，无需重复定义")
print("  4. 所有通过 @save 注册的类都统一管理在 modules 中")

