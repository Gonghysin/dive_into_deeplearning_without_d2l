"""
训练工具函数
"""

import torch


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

