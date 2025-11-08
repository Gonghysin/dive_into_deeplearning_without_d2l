"""
测试 @save 装饰器的使用
演示如何使用通过 @save 注册的模块
"""
import torch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 70)
print("步骤 1: 首先运行 resnet.py 来注册 Residual 类")
print("=" * 70)
print("\n正在导入 resnet.py（这会触发 @save 装饰器的注册）...\n")

# 导入 resnet.py，这会触发 @save 装饰器，将 Residual 注册到 modules
import modern_CNN.resnet

print("\n" + "=" * 70)
print("步骤 2: 查看 modules 中已注册的对象")
print("=" * 70)

from modules import list_registered, get_registry

# 查看所有注册的对象
all_registered = list_registered()
print(f"\n所有已注册对象: {all_registered}")

# 查看详细信息
registry = get_registry()
print("\n详细注册信息:")
for name, info in registry.items():
    print(f"  - {name}: {info['type']} (category: {info['category']})")

print("\n" + "=" * 70)
print("步骤 3: 直接从 modules 导入并使用 Residual 类")
print("=" * 70)

# 现在可以直接从 modules 导入 Residual！
from modules import Residual

print("\n✓ 成功从 modules 导入 Residual 类！")
print(f"  Residual 类型: {type(Residual)}")
print(f"  Residual 位置: {Residual.__module__}")

# 使用 Residual 创建模型
print("\n创建 Residual 实例:")
residual_block = Residual(input_channels=64, num_channels=64)
print(f"  {residual_block}")

# 测试前向传播
X = torch.randn(2, 64, 28, 28)
Y = residual_block(X)
print(f"\n前向传播测试:")
print(f"  输入形状: {X.shape}")
print(f"  输出形状: {Y.shape}")

print("\n" + "=" * 70)
print("步骤 4: 在新的程序中使用（不需要导入 resnet.py）")
print("=" * 70)

print("""
现在你可以在任何 Python 文件中直接使用：

```python
# 方法 1: 只要运行过一次 resnet.py，Residual 就已经注册了
import sys, os
sys.path.insert(0, '/path/to/deep_to_dl')

# 首次使用时，需要先导入一次 resnet 来触发注册
import modern_CNN.resnet  

# 然后就可以直接从 modules 导入使用
from modules import Residual

# 创建残差块
block = Residual(64, 64)
```

```python
# 方法 2: 或者在你的程序开头统一导入
from modules import Residual, ResNet18, print_model_summary

# 直接使用
model = Residual(128, 256, use_1x1conv=True, strides=2)
```
""")

print("\n" + "=" * 70)
print("✓ 演示完成！")
print("=" * 70)
print("\n总结:")
print("  1. 在 resnet.py 中使用 @save 装饰 Residual 类")
print("  2. 运行 resnet.py（或导入它），触发注册")
print("  3. 在任何地方通过 'from modules import Residual' 使用")
print("  4. 无需再次导入 modern_CNN.resnet")
print("=" * 70 + "\n")

