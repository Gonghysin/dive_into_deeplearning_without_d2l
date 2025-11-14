from numpy import rint
import torch

# X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
# H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))


# print(torch.matmul(X, W_xh) + torch.matmul(H, W_hh))

# print(torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0)))

import math
from torch import nn
from torch.nn import functional as F
import d2l_utils as d2l




def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01


    # 隐藏层参数    
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)

def rnn(inputs, state, params):
    # inputs的形状：(时间步数量，批量大小，词表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []

    # X的形状：（批量大小， 词表大小）
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

# 从 d2l_utils 导入 RNN 相关工具
# 这些已经在模块中实现了
from d2l_utils import RNNModelScratch, predict_ch8, grad_clipping, train_epoch_ch8, train_ch8



if __name__ == '__main__':
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)   
    # 独热编码
    X = torch.arange(10).reshape((2, 5))
    print(F.one_hot(X.T, 28).shape)

    # 初始化模型参数
    num_hiddens = 512
    net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params, init_rnn_state, rnn)
    state = net.begin_state(X.shape[0], d2l.try_gpu())
    Y, new_state = net(X.to(d2l.try_gpu()), state)
    print(Y.shape, len(new_state), new_state[0].shape)

    print(predict_ch8('time traveller', 10, net, vocab, d2l.try_gpu()))

    num_epochs, lr = 500,1
    train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())