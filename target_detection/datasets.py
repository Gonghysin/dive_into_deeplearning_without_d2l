"""
Pascal VOC 数据集加载器
"""
import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform


class PascalVOCDataset(Dataset):
    """
    Pascal VOC 数据集类，用于目标检测。
    """

    def __init__(self, data_folder, split, keep_difficult=False):
        """
        初始化数据集。
        
        Args:
            data_folder: 数据文件夹路径，包含 JSON 文件
            split: 'train' 或 'test'，表示训练集或测试集
            keep_difficult: 是否保留标记为 difficult 的目标
        """
        self.split = split.upper()
        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # 读取数据文件
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        """
        获取一个样本。
        
        Args:
            i: 样本索引
            
        Returns:
            image: 图像张量，维度 (3, 224, 224)
            boxes: 边界框张量，维度 (n_objects, 4)，格式 [xmin, ymin, xmax, ymax]
            labels: 类别标签张量，维度 (n_objects,)
            difficulties: 困难标记张量，维度 (n_objects,)
        """
        # 读取图像
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # 读取对象信息
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects,)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects,)

        # 如果不保留困难目标，则过滤掉
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # 应用数据增强/转换
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    def __len__(self):
        """返回数据集大小"""
        return len(self.images)

    def collate_fn(self, batch):
        """
        自定义的 collate 函数，因为每张图像的目标数量不同。
        
        Args:
            batch: 一个 batch 的样本列表
            
        Returns:
            images: 图像张量，维度 (batch_size, 3, 224, 224)
            boxes: 边界框列表，长度为 batch_size
            labels: 标签列表，长度为 batch_size
            difficulties: 困难标记列表，长度为 batch_size
        """
        images = []
        boxes = []
        labels = []
        difficulties = []

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties

