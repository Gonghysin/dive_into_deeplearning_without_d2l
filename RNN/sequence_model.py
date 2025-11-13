"""
时间序列预测：使用简单的多层感知机预测正弦波序列
演示了单步预测、多步预测和k步预测的区别
"""

import os
# 设置环境变量解决 macOS 上的 fork 问题（matplotlib 多进程安全）
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

import torch
from torch import nn
# 导入自定义的工具函数：绘图、数据加载、训练
from d2l_utils import plot, load_array, train


def init_weights(m):
    """初始化网络权重
    
    使用 Xavier 均匀分布初始化线性层的权重
    Xavier 初始化可以保持前向传播和反向传播时方差一致
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def get_net():
    """构建神经网络模型
    
    网络结构：
    - 输入层：4 个特征（tau=4，即前4个时间步）
    - 隐藏层：10 个神经元 + ReLU 激活
    - 输出层：1 个输出（预测下一个时间步的值）
    """
    net = nn.Sequential(
        nn.Linear(4, 10),  # 输入层 -> 隐藏层：4 -> 10
        nn.ReLU(),          # ReLU 激活函数
        nn.Linear(10, 1)    # 隐藏层 -> 输出层：10 -> 1
    )
    net.apply(init_weights)  # 应用权重初始化
    return net


# 定义损失函数：均方误差，不进行 reduction（保留每个样本的损失）
loss = nn.MSELoss(reduction='none')


if __name__ == '__main__':

    # ========== 第一部分：生成时间序列数据 ==========
    T = 1000  # 时间序列长度
    # 生成时间点：1, 2, 3, ..., 1000
    time = torch.arange(1, T+1, dtype=torch.float32)
    # 生成数据：正弦波 + 高斯噪声
    # sin(0.01 * time) 产生周期性变化
    # torch.normal(0, 0.2, (T,)) 添加均值为0、标准差为0.2的噪声
    x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))

    # 绘制原始时间序列数据
    plot(time, [x], xlabel='time', ylabel='x', xlim=[1, 1000], figsize=(6, 3))

    # ========== 第二部分：构造训练数据（特征和标签）==========
    tau = 4  # 时间窗口大小：使用前 4 个时间步预测下一个时间步
    # 创建特征矩阵：(996, 4)
    # 996 = T - tau = 1000 - 4，表示有 996 个训练样本
    # 4 = tau，每个样本有 4 个特征（前4个时间步的值）
    features = torch.zeros((T - tau, tau))
    print(features.shape)  # torch.Size([996, 4])

    # 构造特征矩阵：滑动窗口
    # 第 i 列包含从时间步 i 到 i+(T-tau) 的数据
    for i in range(tau):
        # features[:, 0] = x[0:996]   # 时间步 0 到 995
        # features[:, 1] = x[1:997]   # 时间步 1 到 996
        # features[:, 2] = x[2:998]   # 时间步 2 到 997
        # features[:, 3] = x[3:999]   # 时间步 3 到 998
        features[:, i] = x[i: T - tau + i]
    print(features.shape)  # torch.Size([996, 4])
    print(features)

    # 构造标签：每个样本对应的下一个时间步的真实值
    # labels[i] 对应 features[i] 之后的值，即 x[tau+i]
    # 例如：features[0] = [x[0], x[1], x[2], x[3]]，labels[0] = x[4]
    labels = x[tau:].reshape((-1, 1))  # (996, 1)

    # ========== 第三部分：训练模型 ==========
    batch_size, n_train = 16, 600  # 批量大小16，只使用前600个样本训练
    # 创建数据加载器：只用前 600 个样本训练，其余 396 个样本用于测试
    train_iter = load_array((features[:n_train], labels[:n_train]), 
                           batch_size=batch_size, is_train=True)

    # 创建网络并训练
    net = get_net()
    train(net, train_iter, loss, epochs=5, lr=0.01)

    # ========== 第四部分：单步预测（One-step Prediction）==========
    # 单步预测：每次都使用真实的历史数据来预测下一步
    # 即：用 x[t-4:t] 预测 x[t+1]
    onestep_preds = net(features)  # (996, 1)
    print(onestep_preds.shape)
    
    # 绘制原始数据和单步预测结果
    plot([time, time[tau:]],  # x轴：完整时间 和 从tau开始的时间
        [x.detach().numpy(), onestep_preds.detach().numpy()],  # y轴：原始数据 和 预测
        'time', 'x', 
        legend=['data', '1-step preds'],  # 图例
        xlim=[1, 1000], figsize=(6, 3))
    
    # ========== 第五部分：多步预测（Multi-step Prediction）==========
    # 多步预测：使用模型自己的预测值作为后续预测的输入
    # 这会导致误差累积，预测效果通常比单步预测差
    multistep_preds = torch.zeros(T)
    # 前 n_train + tau 个值直接使用真实数据（训练集 + 初始窗口）
    multistep_preds[: n_train + tau] = x[: n_train + tau]
    
    # 从 n_train + tau 开始逐步预测
    for i in range(n_train + tau, T):
        # 使用前 tau 个预测值来预测下一个值
        # 注意：这里的输入是模型自己之前的预测，不是真实值
        multistep_preds[i] = net(
            multistep_preds[i - tau:i].reshape((1, -1))  # (1, 4)
        )
    
    # 绘制原始数据、单步预测和多步预测的对比
    plot([time, time[tau:], time[n_train + tau:]],  # 三条曲线的 x 轴
        [x.detach().numpy(),                          # 原始数据
         onestep_preds.detach().numpy(),              # 单步预测
         multistep_preds[n_train + tau:].detach().numpy()],  # 多步预测
        'time', 'x', 
        legend=['data', '1-step preds', 'multistep preds'],
        xlim=[1, 1000], figsize=(6, 3))

    # ========== 第六部分：k步预测（k-step-ahead Prediction）==========
    # k步预测：提前预测未来k个时间步
    # 例如：4步预测表示用 x[t-4:t] 直接预测 x[t+4]
    max_steps = 64  # 最大预测步数

    # 创建新的特征矩阵，包含原始特征和未来的预测
    # 形状：(933, 68) = (T - tau - max_steps + 1, tau + max_steps)
    # 933 = 1000 - 4 - 64 + 1，表示有933个起始点
    # 68 = 4 + 64，前4列是真实特征，后64列是预测值
    features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))

    # 填充真实特征（前 tau=4 列）
    for i in range(tau):
        # 每一列包含从不同起始点开始的序列
        features[:, i] = x[i:i + T - tau - max_steps + 1]

    # 逐步预测未来的 max_steps=64 个时间步
    for i in range(tau, tau + max_steps):
        # 使用前 tau 个时间步（可能包含预测值）来预测第 i 列
        # features[:, i - tau:i] 的形状是 (933, 4)
        # net() 输出形状是 (933, 1)
        # squeeze() 将其变为 (933,) 以赋值给 features[:, i]
        features[:, i] = net(features[:, i - tau:i]).squeeze()

    # 选择要展示的预测步数：1步、4步、16步、64步
    steps = (1, 4, 16, 64)
    
    # 绘制不同步数的预测结果
    # 对于每个步数k，绘制从 time[tau+k-1] 开始的预测曲线
    plot(
        [time[tau + i - 1: T - max_steps + i] for i in steps],  # x轴：每个步数对应的时间
        [features[:, (tau + i - 1)].detach().numpy() for i in steps],  # y轴：对应的预测值
        xlabel='time',
        ylabel='x',
        legend=[f'{i}-step preds' for i in steps],  # 图例：1-step, 4-step, 16-step, 64-step
        xlim=[5, 1000], 
        figsize=(6, 3)
    )
