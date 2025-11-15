# TinyDetector - 轻量级目标检测器

基于 VGG16 和 SSD 思想的目标检测模型，用于 Pascal VOC 数据集。

## 📁 项目结构

```
target_detection/
├── model.py              # 模型定义（VGGBase, PredictionConvolutions, TinyDetector, MultiBoxLoss）
├── datasets.py           # 数据集加载器
├── utils.py              # 工具函数
├── train.py              # 训练脚本
├── detect.py             # 检测和可视化脚本
├── create_data_lists.py  # 数据预处理脚本
└── results/              # 结果保存目录
    ├── checkpoints/      # 模型检查点
    ├── plots/            # 训练曲线图
    └── detections/       # 检测结果图
```

## 🚀 快速开始

### 🎯 效果演示（推荐首先运行）

运行测试程序快速查看模型效果：

```bash
python demo_test.py
```

**此程序会自动：**
- ✅ 下载/准备 10 张测试图片
- ✅ 加载模型进行目标检测
- ✅ 生成 10 张带检测框的可视化结果
- ✅ 创建一个汇总展示页面

**结果保存位置：**
- `results/demo_test/result_01.jpg ~ result_10.jpg` - 单张检测结果
- `results/demo_test/summary.jpg` - 汇总展示（推荐查看）
- `test_images/` - 测试图片（可替换为自己的图片）

**自定义测试：**
将您自己的图片放到 `test_images/` 目录，再次运行即可！

---

## 🚀 使用方法

### 1. 准备数据

首先下载 Pascal VOC 2007 和 2012 数据集，然后运行数据预处理脚本：

```bash
python create_data_lists.py
```

### 2. 训练模型

运行训练脚本：

```bash
python train.py
```

训练过程中会：
- 自动进行训练和验证
- 每 10 个 epoch 保存一次检查点
- 保存验证损失最低的最佳模型
- 每 5 个 epoch 更新训练曲线图

**训练结果保存位置：**
- 模型检查点: `results/checkpoints/`
  - `checkpoint_latest.pth` - 最新检查点
  - `checkpoint_epoch_*.pth` - 每 10 个 epoch 的检查点
  - `best_model.pth` - 最佳模型
- 训练曲线: `results/plots/training_curves.png`

### 3. 检测推理

#### 检测单张图像

```bash
python detect.py --checkpoint results/checkpoints/best_model.pth \
                 --image path/to/image.jpg \
                 --output_dir results/detections
```

#### 批量检测

```bash
python detect.py --checkpoint results/checkpoints/best_model.pth \
                 --image_dir path/to/images/ \
                 --output_dir results/detections
```

#### 参数说明

- `--checkpoint`: 模型检查点路径（必需）
- `--image`: 单张图像路径
- `--image_dir`: 图像目录（批量检测）
- `--output_dir`: 输出目录（默认：`results/detections`）
- `--min_score`: 最低置信度阈值（默认：0.2）
- `--max_overlap`: NMS IoU 阈值（默认：0.45）
- `--top_k`: 最多保留的检测框数量（默认：200）

## 📊 训练可视化

训练过程中会自动生成训练曲线图，包括：

1. **损失曲线**: 训练损失和验证损失随 epoch 的变化
2. **学习率曲线**: 学习率的衰减情况

可视化图保存在 `results/plots/training_curves.png`

## 🎯 模型架构

### 网络结构

```
输入图像 (224×224×3)
    ↓
VGGBase (特征提取)
    ↓
特征图 (7×7×512)
    ↓
PredictionConvolutions (预测头)
    ↓
├─ 位置偏移 (441, 4)
└─ 类别分数 (441, 21)
```

### 先验框设计

- 特征图: 7×7
- 每个位置的先验框: 9 个
  - 尺度: [0.2, 0.4, 0.6]
  - 宽高比: [1.0, 2.0, 0.5]
- 总先验框数: 7×7×9 = 441

## 📈 训练参数

- **总 Epoch 数**: 230
- **批次大小**: 32
- **初始学习率**: 0.001
- **学习率衰减**: 
  - 在 epoch 150 和 190 衰减
  - 衰减系数: 0.1
- **优化器**: SGD
  - 动量: 0.9
  - 权重衰减: 5e-4

## 🎨 检测结果

检测结果会显示：
- 彩色边界框（不同类别不同颜色）
- 类别标签和置信度分数

所有检测结果保存在 `results/detections/` 目录。

## 📝 注意事项

1. **数据路径**: 修改 `train.py` 中的 `data_folder` 为你的数据集路径
2. **显存要求**: 默认批次大小为 32，如果显存不足可以减小
3. **训练时间**: 完整训练 230 个 epoch 需要较长时间，建议使用 GPU
4. **中文字体**: 可视化使用 Hiragino Sans GB 字体，如需更改请修改对应文件

## 🔧 自定义配置

### 修改训练参数

编辑 `train.py` 中的参数：

```python
total_epochs = 230      # 总训练轮数
batch_size = 32         # 批次大小
lr = 1e-3              # 学习率
decay_lr_at = [150, 190]  # 学习率衰减的 epoch
```

### 修改检测阈值

在推理时调整参数：

```bash
python detect.py --min_score 0.5 --max_overlap 0.3
```

## 📚 参考

- 模型设计参考 SSD (Single Shot MultiBox Detector)
- 骨干网络: VGG16
- 损失函数: MultiBox Loss (定位损失 + 分类损失)
- 难负样本挖掘 (Hard Negative Mining)

---

**作者**: TinyDetector 项目
**最后更新**: 2025

