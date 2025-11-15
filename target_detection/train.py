"""
目标检测训练脚本 - TinyDetector
包含训练、验证、可视化和模型保存功能
"""
import time
import os
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from matplotlib.font_manager import FontProperties
from datasets import PascalVOCDataset
from model import TinyDetector, MultiBoxLoss
from utils import *

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']
plt.rcParams['axes.unicode_minus'] = False

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# ==================== 数据参数 ====================
data_folder = '/root/autodl-tmp/dive_into_deeplearning_without_d2l/data/VOCdevkit'
keep_difficult = True  # 是否使用标记为困难的目标
n_classes = len(label_map)  # 类别数量

# ==================== 训练参数 ====================
total_epochs = 230  # 总训练轮数
batch_size = 32  # 批次大小
workers = 4  # 数据加载的工作线程数
print_freq = 100  # 每多少个 batch 打印一次训练状态
lr = 1e-3  # 初始学习率
decay_lr_at = [150, 190]  # 在这些 epoch 降低学习率
decay_lr_to = 0.1  # 学习率衰减系数
momentum = 0.9  # SGD 动量
weight_decay = 5e-4  # 权重衰减

# ==================== 结果保存路径 ====================
results_dir = '/Users/mac/PycharmProjects/my/my_learn/deep_to_dl/target_detection/results'
checkpoint_dir = os.path.join(results_dir, 'checkpoints')
plots_dir = os.path.join(results_dir, 'plots')

# 创建目录
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

def train(train_loader, model, criterion, optimizer, epoch):
    """
    训练一个 epoch。
    
    Args:
        train_loader: 训练数据加载器
        model: 模型
        criterion: MultiBox 损失函数
        optimizer: 优化器
        epoch: 当前 epoch 编号
        
    Returns:
        avg_loss: 平均损失
    """
    model.train()  # 训练模式

    batch_time = AverageMeter()  # 批次处理时间
    data_time = AverageMeter()  # 数据加载时间
    losses = AverageMeter()  # 损失

    start = time.time()

    # 遍历批次
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # 将数据移动到设备
        images = images.to(device)  # (N, 3, 224, 224)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # 前向传播
        predicted_locs, predicted_scores = model(images)  # (N, 441, 4), (N, 441, n_classes)

        # 计算损失
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 更新参数
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # 打印训练状态
        if i % print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'批次时间 {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'数据时间 {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'损失 {losses.val:.4f} ({losses.avg:.4f})')

    del predicted_locs, predicted_scores, images, boxes, labels  # 释放内存

    return losses.avg


def validate(val_loader, model, criterion):
    """
    验证模型性能。
    
    Args:
        val_loader: 验证数据加载器
        model: 模型
        criterion: MultiBox 损失函数
        
    Returns:
        avg_loss: 平均验证损失
    """
    model.eval()  # 评估模式

    batch_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    with torch.no_grad():  # 不计算梯度
        for i, (images, boxes, labels, _) in enumerate(val_loader):
            # 将数据移动到设备
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # 前向传播
            predicted_locs, predicted_scores = model(images)

            # 计算损失
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)

            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)

            start = time.time()

    print(f'\n验证: 批次时间 {batch_time.avg:.3f}\t验证损失 {losses.avg:.4f}\n')

    return losses.avg


def plot_training_curves(train_losses, val_losses, learning_rates, save_path):
    """
    绘制训练曲线。
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        learning_rates: 学习率列表
        save_path: 保存路径
    """
    epochs = range(1, len(train_losses) + 1)

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
    if val_losses:
        ax1.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('损失', fontsize=12)
    ax1.set_title('训练和验证损失曲线', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 学习率曲线
    ax2.plot(epochs, learning_rates, 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('学习率', fontsize=12)
    ax2.set_title('学习率变化曲线', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # 使用对数坐标

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'训练曲线已保存到: {save_path}')


def save_checkpoint(epoch, model, optimizer, train_loss, val_loss, save_dir):
    """
    保存模型检查点。
    
    Args:
        epoch: 当前 epoch
        model: 模型
        optimizer: 优化器
        train_loss: 训练损失
        val_loss: 验证损失
        save_dir: 保存目录
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    
    # 保存最新的检查点
    latest_path = os.path.join(save_dir, 'checkpoint_latest.pth')
    torch.save(state, latest_path)
    
    # 每 10 个 epoch 保存一次
    if (epoch + 1) % 10 == 0:
        epoch_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(state, epoch_path)
        print(f'检查点已保存到: {epoch_path}')
    
    return latest_path


def main():
    """
    主训练函数。
    """
    print("=" * 60)
    print("TinyDetector 目标检测训练")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"类别数: {n_classes}")
    print(f"批次大小: {batch_size}")
    print(f"总 Epoch 数: {total_epochs}")
    print(f"结果保存路径: {results_dir}")
    print("=" * 60)

    # 初始化模型和优化器
    print("\n初始化模型...")
    model = TinyDetector(n_classes=n_classes)
    model.to(device)
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 加载训练数据
    print("\n加载训练数据...")
    train_dataset = PascalVOCDataset(
        data_folder,
        split='train',
        keep_difficult=keep_difficult
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=workers,
        pin_memory=True
    )
    
    print(f"训练集大小: {len(train_dataset)}")

    # 加载验证数据
    print("加载验证数据...")
    val_dataset = PascalVOCDataset(
        data_folder,
        split='test',
        keep_difficult=keep_difficult
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=workers,
        pin_memory=True
    )
    
    print(f"验证集大小: {len(val_dataset)}")

    # 用于记录训练过程
    train_losses = []
    val_losses = []
    learning_rates = []
    best_val_loss = float('inf')

    print("\n开始训练...\n")
    
    # 训练循环
    for epoch in range(total_epochs):
        # 学习率衰减
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        # 训练一个 epoch
        train_loss = train(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch
        )
        train_losses.append(train_loss)

        # 验证
        val_loss = validate(
            val_loader=val_loader,
            model=model,
            criterion=criterion
        )
        val_losses.append(val_loss)

        # 打印 epoch 总结
        print(f'Epoch {epoch} 完成 - '
              f'训练损失: {train_loss:.4f}, '
              f'验证损失: {val_loss:.4f}, '
              f'学习率: {current_lr:.6f}')

        # 保存检查点
        save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            train_loss=train_loss,
            val_loss=val_loss,
            save_dir=checkpoint_dir
        )

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f'✓ 最佳模型已保存 (验证损失: {val_loss:.4f})')

        # 每 5 个 epoch 绘制一次训练曲线
        if (epoch + 1) % 5 == 0 or epoch == total_epochs - 1:
            curve_path = os.path.join(plots_dir, 'training_curves.png')
            plot_training_curves(train_losses, val_losses, learning_rates, curve_path)

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"所有结果已保存到: {results_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()