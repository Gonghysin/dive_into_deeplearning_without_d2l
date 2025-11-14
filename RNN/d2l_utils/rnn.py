"""
循环神经网络（RNN）相关工具
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModelScratch:
    """从零开始实现的 RNN 模型
    
    这是一个基础的 RNN 模型框架，需要用户提供参数初始化函数、
    状态初始化函数和前向传播函数。
    
    参数:
        vocab_size: 词表大小
        num_hiddens: 隐藏单元数
        device: 设备（CPU 或 GPU）
        get_params: 参数初始化函数，返回模型参数列表
        init_state: 状态初始化函数，返回初始隐状态
        forward_fn: 前向传播函数
    
    属性:
        vocab_size: 词表大小
        num_hiddens: 隐藏单元数
        params: 模型参数列表
        init_state: 状态初始化函数
        forward_fn: 前向传播函数
    
    示例:
        >>> def get_params(vocab_size, num_hiddens, device):
        ...     # 定义模型参数
        ...     return [W_xh, W_hh, b_h, W_hq, b_q]
        >>> 
        >>> def init_rnn_state(batch_size, num_hiddens, device):
        ...     # 初始化隐状态
        ...     return (torch.zeros((batch_size, num_hiddens), device=device),)
        >>> 
        >>> def rnn(inputs, state, params):
        ...     # RNN 前向传播
        ...     outputs = []
        ...     H, = state
        ...     for X in inputs:
        ...         H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        ...         Y = torch.mm(H, W_hq) + b_q
        ...         outputs.append(Y)
        ...     return torch.cat(outputs, dim=0), (H,)
        >>> 
        >>> net = RNNModelScratch(len(vocab), num_hiddens, device,
        ...                       get_params, init_rnn_state, rnn)
    """
    
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        """初始化 RNN 模型
        
        参数:
            vocab_size: 词表大小
            num_hiddens: 隐藏单元数
            device: 计算设备
            get_params: 获取参数的函数
            init_state: 初始化状态的函数
            forward_fn: 前向传播函数
        """
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        # 初始化模型参数
        self.params = get_params(vocab_size, num_hiddens, device)
        # 保存状态初始化和前向传播函数
        self.init_state, self.forward_fn = init_state, forward_fn
    
    def __call__(self, X, state):
        """前向传播
        
        参数:
            X: 输入序列，形状 (batch_size, num_steps)
            state: 隐状态
        
        返回:
            output: 输出序列
            state: 更新后的隐状态
        """
        # 将输入转换为 one-hot 编码
        # X.T: (num_steps, batch_size)
        # one_hot 后: (num_steps, batch_size, vocab_size)
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)
    
    def begin_state(self, batch_size, device):
        """初始化隐状态
        
        参数:
            batch_size: 批量大小
            device: 设备
        
        返回:
            state: 初始隐状态
        """
        return self.init_state(batch_size, self.num_hiddens, device)


def predict_ch8(prefix, num_preds, net, vocab, device):
    """使用 RNN 模型进行文本预测
    
    给定一个前缀字符串，预测接下来的若干个字符。
    
    参数:
        prefix: 前缀字符串（用于预热模型）
        num_preds: 要预测的字符数量
        net: RNN 模型
        vocab: 词表对象
        device: 计算设备
    
    返回:
        str: 包含前缀和预测结果的完整字符串
    
    工作原理:
        1. 使用前缀字符预热模型（更新隐状态）
        2. 基于当前状态，逐个预测下一个字符
        3. 将预测的字符作为下一次的输入
        4. 重复步骤2-3，直到预测足够数量的字符
    
    示例:
        >>> prefix = 'time traveller'
        >>> prediction = predict_ch8(prefix, 50, net, vocab, device)
        >>> print(prediction)
        time traveller the time traveller smiled the time trav...
    """
    # 初始化隐状态（批量大小为1，因为只预测一个序列）
    state = net.begin_state(batch_size=1, device=device)
    
    # outputs 保存所有输出的索引（从前缀的第一个字符开始）
    outputs = [vocab[prefix[0]]]
    
    # 定义一个函数来获取输入：返回最后一个输出作为下一个输入
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    
    # 第一阶段：使用前缀预热模型
    # 遍历前缀的剩余字符（从第二个字符开始）
    for y in prefix[1:]:
        # 前向传播，更新隐状态
        _, state = net(get_input(), state)
        # 将当前字符的索引添加到输出
        outputs.append(vocab[y])
    
    # 第二阶段：开始预测
    for _ in range(num_preds):
        # 使用当前状态进行预测
        y, state = net(get_input(), state)
        # 选择概率最大的字符（argmax）
        # y 的形状: (1, vocab_size)
        # argmax(dim=1) 找到词表维度上的最大值索引
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    
    # 将索引转换回字符，并拼接成字符串
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def grad_clipping(net, theta):
    """梯度裁剪
    
    将梯度的 L2 范数裁剪到 theta，防止梯度爆炸。
    
    参数:
        net: 神经网络模型（或参数列表）
        theta: 梯度裁剪的阈值
    
    工作原理:
        1. 计算所有参数梯度的 L2 范数
        2. 如果范数大于 theta，按比例缩放所有梯度
        3. 如果范数小于等于 theta，梯度保持不变
    
    示例:
        >>> # 在训练循环中
        >>> loss.backward()
        >>> grad_clipping(net, 1.0)  # 将梯度裁剪到范数 ≤ 1.0
        >>> optimizer.step()
    """
    # 获取所有参数
    if isinstance(net, torch.nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    
    # 计算所有参数梯度的 L2 范数
    # norm = sqrt(sum(||grad||^2))
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    
    # 如果范数超过阈值，进行裁剪
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def sgd(params, lr, batch_size):
    """小批量随机梯度下降
    
    手动实现的 SGD 优化器，用于从零开始实现的模型。
    
    参数:
        params: 参数列表
        lr: 学习率
        batch_size: 批量大小（用于平均梯度）
    
    说明:
        这个函数会就地更新参数，不需要返回值。
    
    示例:
        >>> # 在训练循环中
        >>> loss.backward()
        >>> sgd(net.params, lr=0.01, batch_size=32)
        >>> # 清零梯度
        >>> for param in net.params:
        ...     param.grad.zero_()
    """
    with torch.no_grad():
        for param in params:
            # 更新参数：param = param - lr * grad / batch_size
            param -= lr * param.grad / batch_size
            # 清零梯度
            param.grad.zero_()


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（用于第8章 RNN）
    
    参数:
        net: 神经网络模型
        train_iter: 训练数据迭代器
        loss: 损失函数
        updater: 优化器或更新函数
        device: 计算设备
        use_random_iter: 是否使用随机采样
    
    返回:
        perplexity: 困惑度（perplexity）
        speed: 训练速度（词元/秒）
    
    说明:
        - 困惑度是语言模型常用的评估指标，等于 exp(平均交叉熵损失)
        - 困惑度越低，模型性能越好
        - 使用随机采样时，每个批次都重新初始化隐状态
        - 使用顺序分区时，隐状态在批次之间传递，需要 detach
    
    示例:
        >>> ppl, speed = train_epoch_ch8(net, train_iter, loss, 
        ...                              updater, device, use_random_iter=False)
        >>> print(f'困惑度: {ppl:.2f}, 速度: {speed:.1f} 词元/秒')
    """
    from .train import Accumulator, Timer
    
    state, timer = None, Timer()
    # metric: [总损失, 词元总数, 占位符]
    metric = Accumulator(2)
    
    for X, Y in train_iter:
        # 初始化或重置隐状态
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化 state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # 使用顺序分区时，需要 detach 隐状态
            # 这样反向传播不会穿过整个序列
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # 单个隐状态（非元组）
                state.detach_()
            else:
                # 多个隐状态（元组）
                for s in state:
                    s.detach_()
        
        # y 的形状: (batch_size, num_steps)
        # 转置后变为 (num_steps, batch_size)，然后展平为一维
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        
        # 前向传播
        y_hat, state = net(X, state)
        # 计算损失（平均）
        l = loss(y_hat, y.long()).mean()
        
        # 反向传播和参数更新
        if isinstance(updater, torch.optim.Optimizer):
            # 使用 PyTorch 优化器
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            # 使用自定义更新函数
            l.backward()
            grad_clipping(net, 1)
            # 注意：这里 batch_size=1 是因为损失已经取了平均
            updater(batch_size=1)
        
        # 累积损失和词元数
        metric.add(l * y.numel(), y.numel())
    
    # 计算困惑度和速度
    # 困惑度 = exp(平均损失)
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    """训练模型（用于第8章 RNN）
    
    参数:
        net: 神经网络模型
        train_iter: 训练数据迭代器
        vocab: 词表对象
        lr: 学习率
        num_epochs: 训练轮数
        device: 计算设备
        use_random_iter: 是否使用随机采样（默认 False）
    
    说明:
        - 每隔 10 个 epoch 打印一次预测结果
        - 使用 Animator 实时绘制困惑度曲线
        - 训练结束后展示最终的预测结果
    
    示例:
        >>> batch_size, num_steps = 32, 35
        >>> train_iter, vocab = load_data_time_machine(batch_size, num_steps)
        >>> net = RNNModelScratch(len(vocab), 512, device, 
        ...                       get_params, init_rnn_state, rnn)
        >>> train_ch8(net, train_iter, vocab, lr=1, num_epochs=500, 
        ...          device=device)
    """
    from .plot import Animator
    
    # 定义损失函数
    loss = nn.CrossEntropyLoss()
    # 创建动画器，用于绘制训练曲线
    animator = Animator(xlabel='epoch', ylabel='perplexity', 
                       legend=['train'], xlim=[10, num_epochs])
    
    # 初始化优化器
    if isinstance(net, nn.Module):
        # PyTorch 内置模型使用 PyTorch 优化器
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        # 从零实现的模型使用自定义 SGD
        updater = lambda batch_size: sgd(net.params, lr, batch_size)
    
    # 定义预测函数
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    
    # 训练循环
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter
        )
        # 每 10 个 epoch 打印一次预测结果和更新曲线
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    
    # 训练结束，打印最终结果
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
    
    # 显示动画
    animator.show()

