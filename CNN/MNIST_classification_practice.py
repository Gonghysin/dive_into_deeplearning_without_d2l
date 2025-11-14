import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torch.utils.data import DataLoader   # DataLoader 用于对数据集进行批量加载、打乱和并行加速处理
from torchvision.datasets import mnist
from torchvision import transforms
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']
plt.rcParams['axes.unicode_minus'] = False

# 全连接网络构建
# 在这里，我们构建了输入层、四层全连接层和输出层，输入层的节点个数为784,FC1的节点个数为512,FC2的节点个数为256,FC3的节点个数为128,输出层的节点个数是10（分类10个数）。每个全连接层后都接一个 激活函数，这里激活函数选用Relu。

class Net(nn.Module):
    def __init__(self, in_c = 784, out_c = 10):
        super(Net, self).__init__()

        # 定义全连接层
        self.fc1 = nn.Linear(in_c, 512)
        self.ac1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(512, 256)
        self.ac2 = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(256, 128)
        self.ac3 = nn.ReLU(inplace=True)

        self.fc4 = nn.Linear(128, out_c)
    
    def forward(self, x):
        x = self.ac1(self.fc1(x))
        x = self.ac2(self.fc2(x))
        x = self.ac3(self.fc3(x))
        x = self.fc4(x)
        return x

# 数据加载及网络输入
# pytorch内置集成了MNIST数据集，只需要几行代码就可加载，关于加载的具体方法下一章节会详细解释。

def load_MINST_data(Is_Visualize = False):
    train_set = mnist.MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
    test_set = mnist.MNIST('./data', train=False, transform=transforms.ToTensor(), download=True)
    train_data = DataLoader(train_set, batch_size=64, shuffle=True)
    test_data = DataLoader(test_set, batch_size=128, shuffle=False) 

    # 可视化
    import random
    if Is_Visualize:
        for i in range(4):
            ax = plt.subplot(2, 2, i+1)
            idx = random.randint(0, len(train_set))
            digit_0 = train_set[idx][0].numpy()
            digit_0_image = digit_0.reshape(28, 28)
            ax.imshow(digit_0_image, interpolation="nearest")
            ax.set_title('label: {}'.format(train_set[idx][1]), fontsize=10, color='black')
        plt.show()
    return train_data, test_data

def evaluate_accuracy(data_iter, net, device):
    """计算在指定数据集上模型的准确率"""
    net.eval()  # 将模型设置为评估模式
    acc_sum, n = 0.0, 0
    with torch.no_grad():  # 评估时不需要计算梯度
        for X, y in data_iter:
            X = X.reshape(X.size(0), -1).to(device)
            y = y.to(device)
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def train(train_data, test_data, net, criterion, optimizer, device, num_epochs=20):
    """训练模型"""
    train_losses = []
    train_acces = []
    eval_losses = []
    eval_acces = []
    
    net = net.to(device)
    print("训练设备:", device)
    
    for epoch in range(num_epochs):
        # 训练阶段
        net.train()  # 将模型设置为训练模式
        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
        
        for batch_idx, (X, y) in enumerate(train_data):
            # 数据预处理
            X = X.reshape(X.size(0), -1).to(device)
            y = y.to(device)
            
            # 前向传播
            y_hat = net(X)
            loss = criterion(y_hat, y)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 累计损失和准确率
            train_loss_sum += loss.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
            
            # 定期打印训练信息
            if (batch_idx + 1) % 200 == 0:
                print('[INFO] Epoch-{}-Batch-{}: Train: Loss-{:.4f}, Accuracy-{:.4f}'.format(
                    epoch + 1, batch_idx + 1, loss.item(), 
                    (y_hat.argmax(dim=1) == y).float().mean().item()))
        
        # 记录训练指标
        train_losses.append(train_loss_sum / len(train_data))
        train_acces.append(train_acc_sum / n)
        
        # 评估阶段
        net.eval()
        eval_loss_sum, eval_acc_sum, n_eval = 0.0, 0.0, 0
        
        with torch.no_grad():  # 评估时不需要计算梯度
            for X, y in test_data:
                X = X.reshape(X.size(0), -1).to(device)
                y = y.to(device)
                
                y_hat = net(X)
                loss = criterion(y_hat, y)
                
                eval_loss_sum += loss.item()
                eval_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
                n_eval += y.shape[0]
        
        # 记录评估指标
        eval_losses.append(eval_loss_sum / len(test_data))
        eval_acces.append(eval_acc_sum / n_eval)
        
        print('[INFO] Epoch-{}: Train Loss-{:.4f}, Train Acc-{:.4f} | Eval Loss-{:.4f}, Eval Acc-{:.4f}'.format(
            epoch + 1, train_losses[-1], train_acces[-1], eval_losses[-1], eval_acces[-1]))
    
    return train_losses, train_acces, eval_losses, eval_acces


def visualize_training(losses, acces, eval_losses, eval_acces):
    """可视化训练过程"""
    plt.figure(figsize=(12, 4))
    
    plt.suptitle('训练过程可视化', fontsize=14)
    
    # 损失曲线
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(eval_losses, color='r', label='测试损失', linewidth=2)
    ax1.plot(losses, color='b', label='训练损失', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('损失曲线', fontsize=12, color='black')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(eval_acces, color='r', label='测试准确率', linewidth=2)
    ax2.plot(acces, color='b', label='训练准确率', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('准确率曲线', fontsize=12, color='black')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化网络
    net = Net()
    
    # 加载数据（可选择是否可视化数据样本）
    train_data, test_data = load_MINST_data(Is_Visualize=False)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, weight_decay=5e-4)
    
    # 训练模型
    print("\n开始训练...")
    losses, acces, eval_losses, eval_acces = train(
        train_data, test_data, net, criterion, optimizer, device, num_epochs=20
    )
    
    # 可视化训练过程
    print("\n训练完成，正在生成可视化图表...")
    visualize_training(losses, acces, eval_losses, eval_acces)
    
    # 打印最终结果
    print(f"\n最终结果:")
    print(f"训练集 - 损失: {losses[-1]:.4f}, 准确率: {acces[-1]:.4f}")
    print(f"测试集 - 损失: {eval_losses[-1]:.4f}, 准确率: {eval_acces[-1]:.4f}")

# 训练集 - 损失: 0.0978, 准确率: 0.9718
# 测试集 - 损失: 0.1071, 准确率: 0.9681