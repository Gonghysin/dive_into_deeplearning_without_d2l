"""
目标检测推理和可视化脚本
加载训练好的模型，对图像进行检测，并可视化结果
"""
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model import TinyDetector
from utils import *
import os
import argparse

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_image(image):
    """
    预处理单张图像用于推理。
    
    Args:
        image: PIL 图像
        
    Returns:
        preprocessed_image: 预处理后的张量 (3, 224, 224)
    """
    import torchvision.transforms.functional as FT
    
    # ImageNet 的均值和标准差
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Resize 到 224×224
    resized_image = FT.resize(image, (224, 224))
    
    # 转换为张量
    tensor_image = FT.to_tensor(resized_image)
    
    # 归一化
    normalized_image = FT.normalize(tensor_image, mean=mean, std=std)
    
    return normalized_image


def detect(image_path, model, min_score=0.2, max_overlap=0.45, top_k=200):
    """
    对单张图像进行目标检测。
    
    Args:
        image_path: 图像路径
        model: 检测模型
        min_score: 最低置信度阈值
        max_overlap: NMS 的 IoU 阈值
        top_k: 最多保留的检测框数量
        
    Returns:
        original_image: 原始图像（PIL）
        boxes: 检测框列表
        labels: 类别标签列表
        scores: 置信度分数列表
    """
    # 读取图像
    original_image = Image.open(image_path, mode='r')
    original_image = original_image.convert('RGB')
    
    # 预处理
    image = preprocess_image(original_image)
    image = image.to(device)
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        predicted_locs, predicted_scores = model(image.unsqueeze(0))
    
    # 后处理（NMS等）
    det_boxes, det_labels, det_scores = model.detect_objects(
        predicted_locs,
        predicted_scores,
        min_score=min_score,
        max_overlap=max_overlap,
        top_k=top_k
    )
    
    # 移到 CPU
    det_boxes = det_boxes[0].to('cpu')
    det_labels = det_labels[0].to('cpu')
    det_scores = det_scores[0].to('cpu')
    
    return original_image, det_boxes, det_labels, det_scores


def visualize_detection(image, boxes, labels, scores, save_path=None):
    """
    可视化检测结果。
    
    Args:
        image: PIL 图像
        boxes: 检测框张量 (n_objects, 4)
        labels: 类别标签张量 (n_objects,)
        scores: 置信度分数张量 (n_objects,)
        save_path: 保存路径（可选）
    """
    # 创建图表
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    
    # 获取图像尺寸
    img_width, img_height = image.size
    
    # 绘制每个检测框
    for i in range(boxes.size(0)):
        box = boxes[i].tolist()
        label = labels[i].item()
        score = scores[i].item()
        
        # 转换为像素坐标
        xmin = box[0] * img_width
        ymin = box[1] * img_height
        xmax = box[2] * img_width
        ymax = box[3] * img_height
        
        width = xmax - xmin
        height = ymax - ymin
        
        # 获取类别名称和颜色
        label_name = rev_label_map[label]
        color = label_color_map[label_name]
        
        # 绘制边界框
        rect = patches.Rectangle(
            (xmin, ymin), width, height,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # 绘制标签文本（英文）
        label_text = f'{label_name}: {score:.2f}'
        ax.text(
            xmin, ymin - 5,
            label_text,
            bbox=dict(facecolor=color, alpha=0.5, edgecolor='none', pad=1),
            fontsize=10,
            color='white',
            weight='bold'
        )
    
    ax.axis('off')
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        print(f'Detection result saved to: {save_path}')
    else:
        plt.show()
    
    plt.close()


def batch_detect(image_dir, model, output_dir, min_score=0.2, max_overlap=0.45, top_k=200):
    """
    对一个目录下的所有图像进行批量检测。
    
    Args:
        image_dir: 图像目录
        model: 检测模型
        output_dir: 输出目录
        min_score: 最低置信度阈值
        max_overlap: NMS 的 IoU 阈值
        top_k: 最多保留的检测框数量
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # 获取所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(image_dir) if f.lower().endswith(ext)])
    
    print(f'找到 {len(image_files)} 张图像')
    
    # 逐个检测
    for i, image_file in enumerate(image_files):
        print(f'处理 ({i+1}/{len(image_files)}): {image_file}')
        
        image_path = os.path.join(image_dir, image_file)
        
        try:
            # 检测
            original_image, boxes, labels, scores = detect(
                image_path, model, min_score, max_overlap, top_k
            )
            
            # 可视化并保存
            output_path = os.path.join(output_dir, f'detected_{image_file}')
            visualize_detection(original_image, boxes, labels, scores, output_path)
            
            print(f'  检测到 {boxes.size(0)} 个目标')
        
        except Exception as e:
            print(f'  处理失败: {e}')
    
    print(f'\n批量检测完成！结果保存在: {output_dir}')


def main():
    """
    主函数。
    """
    parser = argparse.ArgumentParser(description='TinyDetector 目标检测推理')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--image', type=str, help='单张图像路径')
    parser.add_argument('--image_dir', type=str, help='图像目录（批量检测）')
    parser.add_argument('--output_dir', type=str, 
                       default='./results/detections',
                       help='输出目录')
    parser.add_argument('--min_score', type=float, default=0.2, help='最低置信度阈值')
    parser.add_argument('--max_overlap', type=float, default=0.45, help='NMS IoU 阈值')
    parser.add_argument('--top_k', type=int, default=200, help='最多保留的检测框数量')
    
    args = parser.parse_args()
    
    # 加载模型
    print(f'加载模型: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = TinyDetector(n_classes=len(label_map))
    
    # 根据检查点的格式加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print('模型加载成功！')
    
    # 单张图像检测
    if args.image:
        print(f'\n检测图像: {args.image}')
        original_image, boxes, labels, scores = detect(
            args.image, model, args.min_score, args.max_overlap, args.top_k
        )
        
        print(f'检测到 {boxes.size(0)} 个目标')
        
        # 保存结果
        os.makedirs(args.output_dir, exist_ok=True)
        image_name = os.path.basename(args.image)
        output_path = os.path.join(args.output_dir, f'detected_{image_name}')
        visualize_detection(original_image, boxes, labels, scores, output_path)
    
    # 批量检测
    elif args.image_dir:
        print(f'\n批量检测目录: {args.image_dir}')
        batch_detect(
            args.image_dir, model, args.output_dir,
            args.min_score, args.max_overlap, args.top_k
        )
    
    else:
        print('请指定 --image 或 --image_dir')


if __name__ == '__main__':
    main()

