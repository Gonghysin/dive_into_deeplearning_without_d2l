"""
训练工具函数
"""

import time
import torch


class Accumulator:
    """在 n 个变量上累加
    
    用于累积训练过程中的各种指标（如损失、准确率等）。
    
    参数:
        n: 要累加的变量数量
    
    示例:
        >>> metric = Accumulator(2)  # 累加损失和样本数
        >>> for batch in data_loader:
        ...     loss, num_samples = compute_loss(batch)
        ...     metric.add(loss, num_samples)
        >>> avg_loss = metric[0] / metric[1]
    """
    def __init__(self, n):
        """初始化累加器
        
        参数:
            n: 累加变量的数量
        """
        self.data = [0.0] * n
    
    def add(self, *args):
        """添加数据到累加器
        
        参数:
            *args: 要累加的值（数量应与初始化时的 n 一致）
        """
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        """重置累加器"""
        self.data = [0.0] * len(self.data)
    
    def __getitem__(self, idx):
        """获取第 idx 个累加值"""
        return self.data[idx]


class Timer:
    """计时器
    
    用于测量代码块的执行时间。
    
    示例:
        >>> timer = Timer()
        >>> # 执行一些操作...
        >>> elapsed_time = timer.stop()
        >>> print(f'耗时: {elapsed_time:.2f} 秒')
    """
    def __init__(self):
        """初始化计时器并开始计时"""
        self.times = []
        self.start()
    
    def start(self):
        """启动计时器"""
        self.tik = time.time()
    
    def stop(self):
        """停止计时器并返回经过的时间（秒）
        
        返回:
            float: 从启动到停止经过的秒数
        """
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    
    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)
    
    def sum(self):
        """返回总时间"""
        return sum(self.times)
    
    def cumsum(self):
        """返回累积时间列表"""
        return [sum(self.times[:i+1]) for i in range(len(self.times))]


def train(net, train_iter, loss, epochs, lr):
    """训练模型
    
    参数:
        net: 神经网络模型
        train_iter: 训练数据迭代器
        loss: 损失函数
        epochs: 训练轮数
        lr: 学习率
    
    示例:
        >>> net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))
        >>> loss = nn.MSELoss()
        >>> train(net, train_iter, loss, epochs=5, lr=0.01)
    """
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        
        print(f'epoch {epoch + 1}, loss: {l.mean().item():.6f}')


def train_with_animator(net, train_iter, loss, epochs, lr, animator=None):
    """训练模型并实时显示损失曲线
    
    参数:
        net: 神经网络模型
        train_iter: 训练数据迭代器
        loss: 损失函数
        epochs: 训练轮数
        lr: 学习率
        animator: Animator 对象（可选），用于绘制训练曲线
    
    示例:
        >>> from d2l_utils import Animator
        >>> net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))
        >>> loss = nn.MSELoss()
        >>> animator = Animator(xlabel='epoch', ylabel='loss', xlim=[1, 10], ylim=[0, 1])
        >>> train_with_animator(net, train_iter, loss, epochs=10, lr=0.01, animator=animator)
    """
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
            
            epoch_loss += l.mean().item()
            num_batches += 1
        
        # 计算平均损失
        avg_loss = epoch_loss / num_batches
        
        # 打印信息
        print(f'epoch {epoch + 1}, loss: {avg_loss:.6f}')
        
        # 更新动画
        if animator is not None:
            animator.add(epoch + 1, avg_loss)
    
    # 显示最终图形
    if animator is not None:
        animator.show()

