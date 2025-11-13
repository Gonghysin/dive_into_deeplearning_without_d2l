import os
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

print("1. 测试导入 torch...")
import torch
print("✓ torch 导入成功")

print("2. 测试创建 tensor...")
x = torch.tensor([1, 2, 3])
print(f"✓ tensor 创建成功: {x}")

print("3. 测试导入 matplotlib...")
import matplotlib
matplotlib.use('TkAgg')
print(f"✓ matplotlib 导入成功，后端: {matplotlib.get_backend()}")

print("4. 测试导入 pyplot...")
import matplotlib.pyplot as plt
print("✓ pyplot 导入成功")

print("5. 测试创建简单图形...")
plt.figure()
plt.plot([1, 2, 3], [1, 2, 3])
print("✓ 创建图形成功")

print("6. 测试显示图形...")
plt.show()
print("✓ 显示图形成功")

print("\n所有测试通过！")

