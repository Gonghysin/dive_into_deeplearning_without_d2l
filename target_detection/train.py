"""
目标检测训练脚本 - TinyDetector
包含训练、验证、可视化和模型保存功能
已优化以充分利用 GPU 性能（支持 RTX 5090 等高端显卡）
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
cudnn.benchmark = True  # 让 cuDNN 自动寻找最优算法
cudnn.deterministic = False  # 提高性能（牺牲一点可复现性）

# ==================== 数据参数 ====================
data_folder = '/root/autodl-tmp/dive_into_deeplearning_without_d2l/data/VOCdevkit'
keep_difficult = True  # 是否使用标记为困难的目标
n_classes = len(label_map)  # 类别数量

# ==================== 训练参数（已优化 GPU 利用率）====================
total_epochs = 230  # 总训练轮数

# GPU 优化配置
# RTX 5090 (32GB): batch_size=128-256
# RTX 4090 (24GB): batch_size=96-128
# RTX 3090 (24GB): batch_size=64-96
batch_size = 128  # 批次大小（从32提升到128）

workers = 8  # 数据加载线程数（从4提升到8）
print_freq = 50  # 打印频率（因batch更大而调整）

lr = 1e-3  # 初始学习率
decay_lr_at = [150, 190]  # 学习率衰减的 epoch
decay_lr_to = 0.1  # 学习率衰减系数
momentum = 0.9  # SGD 动量
weight_decay = 5e-4  # 权重衰减

# 混合精度训练（提升速度 2-3倍，节省显存）
use_amp = True  # 是否使用自动混合精度训练

# ==================== 结果保存路径 ====================
results_dir = './results'  # 结果保存目录（相对路径）
checkpoint_dir = os.path.join(results_dir, 'checkpoints')
plots_dir = os.path.join(results_dir, 'plots')

# 创建目录
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

def train(train_loader, model, criterion, optimizer, epoch, scaler=None):
    """
    训练一个 epoch（支持混合精度）。
    
    Args:
        train_loader: 训练数据加载器
        model: 模型
        criterion: MultiBox 损失函数
        optimizer: 优化器
        epoch: 当前 epoch 编号
        scaler: GradScaler（用于混合精度训练）
        
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

        # 混合精度训练
        if scaler is not None:
            # 使用自动混合精度
            with torch.cuda.amp.autocast():
                # 前向传播
                predicted_locs, predicted_scores = model(images)
                # 计算损失
                loss = criterion(predicted_locs, predicted_scores, boxes, labels)
            
            # 反向传播（混合精度）
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 标准训练
            # 前向传播
            predicted_locs, predicted_scores = model(images)
            # 计算损失
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # 打印训练状态
        if i % print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})')

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

    print(f'\nValidation: Batch Time {batch_time.avg:.3f}\tVal Loss {losses.avg:.4f}\n')

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
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    if val_losses:
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 学习率曲线
    ax2.plot(epochs, learning_rates, 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # 使用对数坐标

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Training curves saved to: {save_path}')


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
        print(f'Checkpoint saved to: {epoch_path}')
    
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
        print(f'Epoch {epoch} Complete - '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'LR: {current_lr:.6f}')

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
            print(f'✓ Best model saved (val loss: {val_loss:.4f})')

        # 每 5 个 epoch 绘制一次训练曲线
        if (epoch + 1) % 5 == 0 or epoch == total_epochs - 1:
            curve_path = os.path.join(plots_dir, 'training_curves.png')
            plot_training_curves(train_losses, val_losses, learning_rates, curve_path)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Results saved to: {results_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()