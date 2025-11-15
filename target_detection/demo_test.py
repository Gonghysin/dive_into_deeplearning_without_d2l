"""
TinyDetector 效果演示测试程序

运行此脚本会：
1. 自动下载/使用测试图片
2. 加载模型进行检测
3. 生成10张带检测框的可视化结果
4. 创建一个汇总展示页面
"""
import torch
import os
import urllib.request
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model import TinyDetector
from detect import preprocess_image
from utils import *
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 测试图片URL（来自公开数据集的示例图片）
SAMPLE_IMAGES = [
    "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=500",  # 狗
    "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=500",  # 猫
    "https://images.unsplash.com/photo-1568572933382-74d440642117?w=500",  # 马
    "https://images.unsplash.com/photo-1601758228041-f3b2795255f1?w=500",  # 狗
    "https://images.unsplash.com/photo-1583511655857-d19b40a7a54e?w=500",  # 鸟
    "https://images.unsplash.com/photo-1530281700549-e82e7bf110d6?w=500",  # 狗
    "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=500",  # 猫
    "https://images.unsplash.com/photo-1553284965-83fd3e82fa5a?w=500",  # 狗
    "https://images.unsplash.com/photo-1588943211346-0908a1fb0b01?w=500",  # 狗
    "https://images.unsplash.com/photo-1592194996308-7b43878e84a6?w=500",  # 猫
]


def download_sample_images(output_dir, num_images=10):
    """
    下载示例图片（如果网络不可用，则使用本地图片）。
    
    Args:
        output_dir: 输出目录
        num_images: 图片数量
        
    Returns:
        image_paths: 图片路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    image_paths = []
    
    print(f"准备测试图片...")
    
    for i in range(num_images):
        image_path = os.path.join(output_dir, f'test_image_{i+1}.jpg')
        
        # 如果本地已有图片，跳过下载
        if os.path.exists(image_path):
            print(f"  [{i+1}/{num_images}] 使用已存在的图片: {image_path}")
            image_paths.append(image_path)
            continue
        
        # 尝试下载图片
        try:
            print(f"  [{i+1}/{num_images}] 下载图片...")
            urllib.request.urlretrieve(SAMPLE_IMAGES[i], image_path)
            image_paths.append(image_path)
        except Exception as e:
            print(f"  下载失败: {e}")
            # 创建一个占位图片
            create_placeholder_image(image_path, i+1)
            image_paths.append(image_path)
    
    print(f"✓ 准备完成，共 {len(image_paths)} 张图片\n")
    return image_paths


def create_placeholder_image(path, number):
    """
    创建占位图片（当无法下载时使用）。
    
    Args:
        path: 保存路径
        number: 图片编号
    """
    img = Image.new('RGB', (500, 500), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # 绘制大号文字
    text = f"测试图片 {number}\n\n请放置您自己的\n测试图片到:\ntarget_detection/\ntest_images/"
    
    # 计算文本位置（居中）
    bbox = draw.textbbox((0, 0), text, font=None)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (500 - text_width) // 2
    y = (500 - text_height) // 2
    
    draw.text((x, y), text, fill=(100, 100, 100), align='center')
    img.save(path)


def detect_single_image(image_path, model, min_score=0.2, max_overlap=0.45, top_k=200):
    """
    对单张图像进行检测。
    
    Args:
        image_path: 图像路径
        model: 模型
        min_score: 最低置信度
        max_overlap: NMS阈值
        top_k: 最多检测数
        
    Returns:
        original_image, boxes, labels, scores
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
    
    # 后处理
    det_boxes, det_labels, det_scores = model.detect_objects(
        predicted_locs, predicted_scores, min_score, max_overlap, top_k
    )
    
    # 移到CPU
    det_boxes = det_boxes[0].to('cpu') if len(det_boxes[0]) > 0 else torch.FloatTensor([])
    det_labels = det_labels[0].to('cpu') if len(det_labels[0]) > 0 else torch.LongTensor([])
    det_scores = det_scores[0].to('cpu') if len(det_scores[0]) > 0 else torch.FloatTensor([])
    
    return original_image, det_boxes, det_labels, det_scores


def visualize_single_detection(image, boxes, labels, scores, save_path, image_name):
    """
    可视化单张图像的检测结果。
    
    Args:
        image: PIL图像
        boxes: 检测框
        labels: 标签
        scores: 分数
        save_path: 保存路径
        image_name: 图像名称
    """
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(image)
    
    img_width, img_height = image.size
    n_objects = boxes.size(0)
    
    # 绘制检测框
    for i in range(n_objects):
        box = boxes[i].tolist()
        label = labels[i].item()
        score = scores[i].item()
        
        # 转换坐标
        xmin = box[0] * img_width
        ymin = box[1] * img_height
        xmax = box[2] * img_width
        ymax = box[3] * img_height
        
        width = xmax - xmin
        height = ymax - ymin
        
        # 获取类别和颜色
        label_name = rev_label_map[label]
        color = label_color_map[label_name]
        
        # 绘制框
        rect = patches.Rectangle(
            (xmin, ymin), width, height,
            linewidth=3, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # 绘制标签（英文）
        label_text = f'{label_name}: {score:.2f}'
        ax.text(
            xmin, ymin - 8,
            label_text,
            bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=3),
            fontsize=11,
            color='white',
            weight='bold'
        )
    
    # 添加标题（英文）
    title = f'{image_name}\nDetected {n_objects} objects'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def create_summary_visualization(result_paths, output_path):
    """
    创建汇总可视化展示。
    
    Args:
        result_paths: 结果图片路径列表
        output_path: 输出路径
    """
    n_images = len(result_paths)
    cols = 2
    rows = (n_images + cols - 1) // cols
    
    fig = plt.figure(figsize=(16, 8 * rows))
    
    for i, img_path in enumerate(result_paths):
        ax = plt.subplot(rows, cols, i + 1)
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')
    
    plt.suptitle('TinyDetector Object Detection Demo', 
                 fontsize=20, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Summary visualization saved: {output_path}")


def main():
    """
    主测试函数。
    """
    print("=" * 70)
    print(" " * 15 + "TinyDetector 效果演示测试")
    print("=" * 70)
    print(f"设备: {device}")
    print("=" * 70)
    
    # 设置路径
    base_dir = '.'  # 当前目录
    test_images_dir = os.path.join(base_dir, 'test_images')
    demo_results_dir = os.path.join(base_dir, 'results', 'demo_test')
    checkpoint_path = os.path.join(base_dir, 'results', 'checkpoints', 'best_model.pth')
    
    os.makedirs(demo_results_dir, exist_ok=True)
    
    # 步骤1: 准备测试图片
    print("\n步骤 1/3: 准备测试图片")
    print("-" * 70)
    image_paths = download_sample_images(test_images_dir, num_images=10)
    
    # 步骤2: 加载模型
    print("\n步骤 2/3: 加载模型")
    print("-" * 70)
    
    model = TinyDetector(n_classes=len(label_map))
    
    # 尝试加载训练好的模型
    if os.path.exists(checkpoint_path):
        print(f"加载训练好的模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✓ 模型加载成功")
    else:
        print("⚠ 未找到训练好的模型，使用预训练的VGG权重初始化")
        print(f"  期望路径: {checkpoint_path}")
        print("  注意: 检测效果可能不佳，建议先训练模型")
    
    model = model.to(device)
    model.eval()
    
    # 步骤3: 进行检测
    print("\n步骤 3/3: 执行目标检测")
    print("-" * 70)
    
    result_paths = []
    detection_stats = []
    
    for i, img_path in enumerate(image_paths):
        image_name = os.path.basename(img_path)
        print(f"[{i+1}/10] 检测: {image_name}... ", end='', flush=True)
        
        try:
            # 检测
            original_image, boxes, labels, scores = detect_single_image(
                img_path, model, min_score=0.3, max_overlap=0.45, top_k=200
            )
            
            n_detections = boxes.size(0)
            print(f"检测到 {n_detections} 个目标")
            
            # 可视化并保存
            result_path = os.path.join(demo_results_dir, f'result_{i+1:02d}.jpg')
            visualize_single_detection(
                original_image, boxes, labels, scores, 
                result_path, f'Test Image {i+1}'
            )
            
            result_paths.append(result_path)
            
            # 统计信息
            if n_detections > 0:
                detected_classes = [rev_label_map[l.item()] for l in labels]
                avg_score = scores.mean().item()
                detection_stats.append({
                    'image': image_name,
                    'n_objects': n_detections,
                    'classes': detected_classes,
                    'avg_score': avg_score
                })
            else:
                detection_stats.append({
                    'image': image_name,
                    'n_objects': 0,
                    'classes': [],
                    'avg_score': 0.0
                })
        
        except Exception as e:
            print(f"失败: {e}")
    
    # 创建汇总展示
    print("\n" + "-" * 70)
    print("生成汇总展示...")
    summary_path = os.path.join(demo_results_dir, 'summary.jpg')
    create_summary_visualization(result_paths, summary_path)
    
    # 打印统计信息
    print("\n" + "=" * 70)
    print("检测统计")
    print("=" * 70)
    
    total_detections = sum(s['n_objects'] for s in detection_stats)
    images_with_detections = sum(1 for s in detection_stats if s['n_objects'] > 0)
    
    print(f"总检测数: {total_detections}")
    print(f"有检测的图片: {images_with_detections}/10")
    
    if total_detections > 0:
        print("\n详细信息:")
        for i, stat in enumerate(detection_stats):
            if stat['n_objects'] > 0:
                classes_str = ', '.join(stat['classes'])
                print(f"  [{i+1}] {stat['image']}: "
                      f"{stat['n_objects']}个目标 ({classes_str}) "
                      f"平均置信度: {stat['avg_score']:.3f}")
    
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
    print(f"\n结果保存位置:")
    print(f"  - 单张检测结果: {demo_results_dir}/")
    print(f"  - 汇总展示: {summary_path}")
    print(f"  - 测试图片: {test_images_dir}/")
    print("\n提示:")
    print("  1. 您可以将自己的测试图片放到 test_images/ 目录")
    print("  2. 重新运行此脚本即可看到检测效果")
    print("  3. 调整 min_score 参数可以控制检测灵敏度")
    print("=" * 70)


if __name__ == '__main__':
    main()

