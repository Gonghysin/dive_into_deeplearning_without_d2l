"""
测试设备检测功能
演示如何使用优化后的 modules 包进行跨平台GPU训练
"""

import torch
from modules import try_gpu, get_device_info

def test_device_detection():
    """测试设备检测功能"""
    print("=" * 80)
    print("设备检测测试")
    print("=" * 80)
    
    # 1. 获取详细的设备信息
    print("\n1. 获取系统设备信息:")
    print("-" * 80)
    device_info = get_device_info()
    print(f"操作系统: {device_info['platform']}")
    print(f"架构: {device_info['machine']}")
    print(f"MPS可用: {device_info['has_mps']}")
    print(f"CUDA可用: {device_info['has_cuda']}")
    
    if device_info['has_cuda']:
        print(f"CUDA设备数量: {device_info['cuda_count']}")
        for cuda_dev in device_info['cuda_devices']:
            print(f"  - GPU {cuda_dev['id']}: {cuda_dev['name']}")
            print(f"    显存: {cuda_dev['total_memory']}")
            print(f"    计算能力: {cuda_dev['capability']}")
    
    # 2. 自动选择最佳设备（带详细输出）
    print("\n2. 自动选择最佳计算设备:")
    print("-" * 80)
    device = try_gpu(verbose=True)
    
    # 3. 测试设备是否可用
    print("\n3. 测试设备:")
    print("-" * 80)
    print(f"选定设备: {device}")
    
    # 创建一个简单的张量测试
    try:
        x = torch.randn(3, 3).to(device)
        y = torch.randn(3, 3).to(device)
        z = x @ y
        print(f"✓ 张量运算测试成功！")
        print(f"  结果形状: {z.shape}")
        print(f"  结果设备: {z.device}")
    except Exception as e:
        print(f"✗ 张量运算测试失败: {e}")
    
    # 4. 静默模式（不打印详细信息）
    print("\n4. 静默模式获取设备:")
    print("-" * 80)
    device_silent = try_gpu(verbose=False)
    print(f"静默模式返回设备: {device_silent}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == '__main__':
    test_device_detection()
