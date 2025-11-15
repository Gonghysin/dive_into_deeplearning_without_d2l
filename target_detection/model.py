from torch import nn
from utils import *
import torch.nn.functional as F
from math import sqrt
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class VGGBase(nn.Module):
    """
    VGG16 基础网络，按照 PyTorch 最佳实践使用层和块的形式重构。
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # VGG16 架构配置：每个块的卷积层数和通道数
        # 格式：(输出通道数, 卷积层数)
        vgg_config = [
            (64, 2),   # Block 1: 224->112
            (128, 2),  # Block 2: 112->56
            (256, 3),  # Block 3: 56->28
            (512, 3),  # Block 4: 28->14
            (512, 3),  # Block 5: 14->7
        ]
        
        # 构建特征提取层
        self.features = self._make_layers(vgg_config)
        
        # 加载预训练权重
        self.load_pretrained_layers()
    
    def _make_conv_block(self, in_channels, out_channels, num_convs):
        """
        创建一个 VGG 卷积块。
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            num_convs: 卷积层数量
            
        Returns:
            nn.Sequential: 包含多个卷积层和 ReLU 激活的序列
        """
        layers = []
        for i in range(num_convs):
            layers.extend([
                nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1
                ),
                nn.ReLU(inplace=True)
            ])
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)
    
    def _make_layers(self, vgg_config):
        """
        根据配置构建完整的 VGG 特征提取层。
        
        Args:
            vgg_config: VGG 配置列表
            
        Returns:
            nn.Sequential: 完整的特征提取网络
        """
        layers = []
        in_channels = 3
        
        for out_channels, num_convs in vgg_config:
            layers.append(self._make_conv_block(in_channels, out_channels, num_convs))
            in_channels = out_channels
        
        return nn.Sequential(*layers)

    def forward(self, image):
        """
        前向传播。

        Args:
            image: 输入图像张量，维度为 (N, 3, 224, 224)
            
        Returns:
            特征图张量，维度为 (N, 512, 7, 7)
        """
        return self.features(image)

    def load_pretrained_layers(self):
        """
        从 torchvision 的预训练 VGG16 模型加载权重。
        """
        try:
            # 加载预训练的 VGG16 模型（使用新的API以避免警告）
            try:
                # 尝试使用新版本的API
                from torchvision.models import VGG16_Weights
                pretrained_vgg = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            except (ImportError, AttributeError):
                # 如果新API不可用，使用旧API
                pretrained_vgg = torchvision.models.vgg16(pretrained=True)
            
            pretrained_features = pretrained_vgg.features
            
            # 获取当前模型的状态字典
            model_dict = self.state_dict()
            
            # 创建权重映射
            pretrained_dict = {}
            pretrained_state = pretrained_features.state_dict()
            
            # 匹配层名称并复制权重
            for pretrained_name, pretrained_param in pretrained_state.items():
                # 预训练模型的层名如: "0.weight", "0.bias", "2.weight" 等
                # 我们的模型层名如: "features.0.0.weight", "features.0.1.weight" 等
                
                # 遍历我们模型的所有层
                for model_name, model_param in model_dict.items():
                    # 检查是否匹配（形状和名称后缀）
                    if (model_name.startswith('features.') and 
                        model_name.endswith(pretrained_name.split('.')[-1]) and
                        model_param.shape == pretrained_param.shape):
                        # 检查是否已经有匹配的权重
                        if model_name not in pretrained_dict:
                            pretrained_dict[model_name] = pretrained_param
                            break
            
            # 如果匹配失败，尝试直接按索引匹配
            if len(pretrained_dict) == 0:
                print("  尝试按层索引匹配...")
                pretrained_layers = list(pretrained_features.children())
                our_blocks = list(self.features.children())
                
                loaded_count = 0
                for our_block_idx, our_block in enumerate(our_blocks):
                    # 遍历每个块中的层
                    for layer in our_block:
                        if isinstance(layer, nn.Conv2d):
                            # 找对应的预训练卷积层
                            for pretrained_layer in pretrained_layers:
                                if isinstance(pretrained_layer, nn.Conv2d):
                                    if (pretrained_layer.weight.shape == layer.weight.shape and
                                        pretrained_layer not in [pl for pl in pretrained_layers[:loaded_count]]):
                                        layer.weight.data = pretrained_layer.weight.data.clone()
                                        if layer.bias is not None and pretrained_layer.bias is not None:
                                            layer.bias.data = pretrained_layer.bias.data.clone()
                                        loaded_count += 1
                                        break
                
                if loaded_count > 0:
                    print(f"✓ 成功加载预训练权重：{loaded_count} 个卷积层")
                    return
            
            # 更新模型权重
            if len(pretrained_dict) > 0:
                model_dict.update(pretrained_dict)
                self.load_state_dict(model_dict)
                print(f"✓ 成功加载预训练权重：{len(pretrained_dict)} 层")
            else:
                print("⚠ 未能加载预训练权重，将使用随机初始化")
                
        except Exception as e:
            print(f"⚠ 加载预训练权重时出错: {e}")
            print("  将使用随机初始化")

class PredictionConvolutions(nn.Module):
    """
    目标检测预测头，用于预测边界框位置和类别置信度。
    
    对于 7×7 的特征图，每个位置有 9 个先验框（3种尺度 × 3种宽高比），
    总共产生 7×7×9 = 441 个预测框。
    """

    def __init__(self, n_classes) -> None:
        """
        初始化预测卷积层。
        
        Args:
            n_classes: 目标类别数量
        """
        super().__init__()

        self.n_classes = n_classes

        # 每个特征图位置有 9 个先验框（3种尺度 × 3种宽高比）
        n_boxes = 9
        
        # 位置预测卷积层：输出 9×4 = 36 个通道
        # 每个先验框预测 4 个值：(Δx, Δy, Δw, Δh) - 边界框的偏移量
        self.loc_conv = nn.Conv2d(512, n_boxes * 4, kernel_size=3, padding=1)
        
        # 类别预测卷积层：输出 9×n_classes 个通道
        # 每个先验框预测每个类别的置信度分数（logits）
        self.c1_conv = nn.Conv2d(512, n_boxes * n_classes, kernel_size=3, padding=1)

        # 初始化卷积层权重
        self.init_conv2d()

    def init_conv2d(self):
        """
        使用 Xavier 均匀分布初始化卷积层权重，保证训练初期梯度稳定。
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, pool5_feats):
        """
        前向传播，预测边界框位置和类别分数。
        
        Args:
            pool5_feats: VGG 输出的特征图，维度 (N, 512, 7, 7)
            
        Returns:
            locs: 边界框位置偏移，维度 (N, 441, 4)
                  441 = 7×7×9，表示所有先验框的位置预测
            classes_scores: 类别置信度分数，维度 (N, 441, n_classes)
                           每个先验框对每个类别的原始分数（logits）
        """
        batch_size = pool5_feats.size(0)

        # ==================== 位置预测 ====================
        # 1. 卷积预测位置偏移
        l_conv = self.loc_conv(pool5_feats)  # (N, 36, 7, 7)
        
        # 2. 调整维度顺序：将通道维度移到最后
        l_conv = l_conv.permute(0, 2, 3, 1).contiguous()  # (N, 7, 7, 36)
        
        # 3. 重塑为 (N, 441个框, 4个坐标)
        locs = l_conv.view(batch_size, -1, 4)  # (N, 441, 4)

        # ==================== 类别预测 ====================
        # 1. 卷积预测类别分数
        c_conv = self.c1_conv(pool5_feats)  # (N, 9×n_classes, 7, 7)
        
        # 2. 调整维度顺序：将通道维度移到最后
        c_conv = c_conv.permute(0, 2, 3, 1).contiguous()  # (N, 7, 7, 9×n_classes)
        
        # 3. 重塑为 (N, 441个框, n_classes个类别分数)
        classes_scores = c_conv.view(batch_size, -1, self.n_classes)  # (N, 441, n_classes)

        return locs, classes_scores

class TinyDetector(nn.Module):
    """
    轻量级目标检测器，整合了特征提取、边界框预测和后处理三大模块。
    
    网络结构：
        - VGGBase: 特征提取骨干网络（输出 7×7×512 特征图）
        - PredictionConvolutions: 预测头（预测 441 个先验框的位置和类别）
        - 先验框机制: 每个特征图位置 9 个框（3种尺度 × 3种宽高比）
    """

    def __init__(self, n_classes, *args, **kwargs) -> None:
        """
        初始化目标检测器。
        
        Args:
            n_classes: 目标类别数量（包括背景类）
        """
        super().__init__(*args, **kwargs)

        self.n_classes = n_classes

        # 特征提取骨干网络（VGG16）
        self.base = VGGBase()
        
        # 预测头（边界框位置 + 类别分数）
        self.pred_convs = PredictionConvolutions(n_classes)

        # 生成先验框（Anchor Boxes），维度 (441, 4)，格式 [cx, cy, w, h]
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image):
        """
        前向传播（训练阶段）。
        
        Args:
            image: 输入图像张量，维度 (N, 3, 224, 224)
            
        Returns:
            locs: 预测的边界框位置偏移，维度 (N, 441, 4)
            classes_scores: 预测的类别分数（logits），维度 (N, 441, n_classes)
        """
        # Step 1: 特征提取
        pool5_feats = self.base(image)  # (N, 512, 7, 7)
        
        # Step 2: 边界框和类别预测
        locs, classes_scores = self.pred_convs(pool5_feats)  # (N, 441, 4), (N, 441, n_classes)

        return locs, classes_scores

    def create_prior_boxes(self):
        """
        生成先验框（Anchor Boxes）。
        
        先验框是预定义的边界框，用于目标检测的基准。对于 7×7 特征图：
        - 每个网格位置生成 9 个先验框（3种尺度 × 3种宽高比）
        - 总共生成 7×7×9 = 441 个先验框
        
        先验框参数：
        - 尺度（Scale）: [0.2, 0.4, 0.6] - 相对于图像大小的比例
        - 宽高比（Aspect Ratio）: [1.0, 2.0, 0.5] - 正方形、横向、纵向
        
        Returns:
            prior_boxes: 先验框张量，维度 (441, 4)
                        格式：[cx, cy, w, h]（中心点坐标和宽高，归一化到 0~1）
        """
        # 特征图尺寸 7×7
        fmap_dims = 7
        
        # 先验框的尺度（相对于图像大小）
        obj_scales = [0.2, 0.4, 0.6]
        
        # 先验框的宽高比
        aspect_ratios = [1., 2., 0.5]  # 正方形、横向矩形、纵向矩形

        prior_boxes = []
        
        # 遍历特征图的每个网格位置
        for i in range(fmap_dims):  # 行索引
            for j in range(fmap_dims):  # 列索引
                # 计算网格中心点坐标（归一化到 0~1）
                cx = (j + 0.5) / fmap_dims  # x 坐标
                cy = (i + 0.5) / fmap_dims  # y 坐标

                # 为当前网格位置生成 9 个不同尺度和宽高比的先验框
                for obj_scale in obj_scales:
                    for ratio in aspect_ratios:
                        # 根据宽高比调整宽度和高度
                        # 宽度：scale × sqrt(ratio)
                        # 高度：scale / sqrt(ratio)
                        # 这样保证面积 = scale²
                        prior_boxes.append([
                            cx, 
                            cy, 
                            obj_scale * sqrt(ratio),      # 宽度
                            obj_scale / sqrt(ratio)       # 高度
                        ])

        # 转换为张量并移动到设备（GPU/CPU）
        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (441, 4)
        
        # 确保所有坐标值在 0~1 范围内
        prior_boxes.clamp_(0, 1)
        
        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        目标检测后处理（推理阶段），包含解码、过滤和非极大值抑制（NMS）。
        
        处理流程：
        1. 解码边界框：将预测的偏移量转换为实际坐标
        2. 置信度过滤：过滤掉低于最低分数阈值的检测
        3. NMS（非极大值抑制）：去除重复检测
        4. Top-K 选择：保留分数最高的 K 个检测
        
        Args:
            predicted_locs: 预测的边界框偏移，维度 (N, 441, 4)
            predicted_scores: 预测的类别分数（logits），维度 (N, 441, n_classes)
            min_score: 最低置信度阈值（如 0.5）
            max_overlap: NMS 的 IoU 阈值（如 0.45），超过该值的重叠框将被抑制
            top_k: 每张图像最多保留的检测框数量（如 200）
            
        Returns:
            all_images_boxes: 每张图像的检测框列表，长度为 batch_size
                             每个元素维度 (n_objects, 4)，格式 [xmin, ymin, xmax, ymax]
            all_images_labels: 每张图像的类别标签列表，长度为 batch_size
                              每个元素维度 (n_objects,)
            all_images_scores: 每张图像的置信度分数列表，长度为 batch_size
                              每个元素维度 (n_objects,)
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)  # 441
        
        # 将 logits 转换为概率分布（在类别维度上进行 softmax）
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 441, n_classes)

        # 存储所有图像的检测结果
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        # 确保先验框数量与预测数量一致
        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        # 逐张图像处理
        for i in range(batch_size):
            # ==================== Step 1: 解码边界框 ====================
            # 将预测的偏移量转换为实际的边界框坐标
            # gcxgcy_to_cxcy: 偏移 → 中心点格式 [cx, cy, w, h]
            # cxcy_to_xy: 中心点格式 → 角点格式 [xmin, ymin, xmax, ymax]
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy)
            )  # (441, 4)

            # 当前图像的检测结果（将在后续步骤中填充）
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            # 获取每个先验框的最高分数和对应类别（用于调试，未使用）
            max_scores, best_label = predicted_scores[i].max(dim=1)  # (441,), (441,)

            # ==================== Step 2: 按类别处理（跳过背景类 0）====================
            for c in range(1, self.n_classes):
                # 获取当前类别的所有分数
                class_scores = predicted_scores[i][:, c]  # (441,)
                
                # ==================== Step 3: 置信度过滤 ====================
                # 找出分数高于阈值的检测
                score_above_min_score = class_scores > min_score  # (441,) 布尔张量
                n_above_min_score = score_above_min_score.sum().item()

                # 如果没有高置信度检测，跳过当前类别
                if n_above_min_score == 0:
                    continue

                # 保留高置信度的检测
                class_scores = class_scores[score_above_min_score]  # (n_qualified,)
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # ==================== Step 4: 按分数排序 ====================
                # 按置信度从高到低排序（为 NMS 做准备）
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_qualified, 4)

                # ==================== Step 5: NMS（非极大值抑制）====================
                # 计算所有框之间的 IoU（Intersection over Union）
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_qualified)

                # 初始化抑制标记（0=保留，1=抑制）
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)

                # NMS 主循环：遍历每个框（按分数从高到低）
                for box in range(class_decoded_locs.size(0)):
                    # 如果当前框已被抑制，跳过
                    if suppress[box] == 1:
                        continue

                    # 抑制与当前框 IoU 过高的所有框（保留当前框本身）
                    # overlap[box] > max_overlap 生成布尔张量，标记需要抑制的框
                    suppress = torch.max(suppress, (overlap[box] > max_overlap).to(torch.uint8))

                    # 确保当前框不被抑制
                    suppress[box] = 0

                # ==================== Step 6: 收集保留的检测 ====================
                # 保留未被抑制的框（1 - suppress 将 0→1, 1→0）
                image_boxes.append(class_decoded_locs[1 - suppress])
                # 为保留的框创建类别标签
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                # 保留对应的置信度分数
                image_scores.append(class_scores[1 - suppress])

            # ==================== Step 7: 合并所有类别的检测 ====================
            # 如果没有检测到任何对象，这里可能会出错，需要在实际使用中添加判断
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects,)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects,)
            n_objects = image_scores.size(0)

            # ==================== Step 8: Top-K 选择 ====================
            # 如果检测数量超过 top_k，只保留分数最高的 top_k 个
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k,)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k,)

            # 保存当前图像的检测结果
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores


class MultiBoxLoss(nn.Module):
    """
    MultiBox 损失函数，用于目标检测训练（遵循 SSD 论文定义）。
    
    损失函数由两部分组成：
    1. 定位损失（Localization Loss）：预测边界框位置的准确性
    2. 置信度损失（Confidence Loss）：预测类别的准确性
    
    采用难负样本挖掘（Hard Negative Mining）来平衡正负样本。
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        """
        初始化 MultiBox 损失函数。
        
        Args:
            priors_cxcy: 先验框，维度 (441, 4)，格式 [cx, cy, w, h]
            threshold: IoU 阈值，用于判断先验框是否匹配真实目标（默认 0.5）
            neg_pos_ratio: 负样本与正样本的比例（默认 3:1）
            alpha: 定位损失的权重系数（默认 1.0）
        """
        super(MultiBoxLoss, self).__init__()
        
        self.priors_cxcy = priors_cxcy  # 中心点格式的先验框
        self.priors_xy = cxcy_to_xy(priors_cxcy)  # 转换为角点格式 [xmin, ymin, xmax, ymax]
        self.threshold = threshold  # IoU 匹配阈值
        self.neg_pos_ratio = neg_pos_ratio  # 负正样本比例
        self.alpha = alpha  # 定位损失权重
        
        # 定义损失函数
        self.smooth_l1 = nn.L1Loss()  # 定位损失使用 Smooth L1
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)  # 分类损失使用交叉熵

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        计算 MultiBox 损失。
        
        Args:
            predicted_locs: 预测的边界框偏移，维度 (N, 441, 4)
            predicted_scores: 预测的类别分数，维度 (N, 441, n_classes)
            boxes: 真实边界框列表，长度为 N，每个元素维度 (n_objects, 4)
            labels: 真实类别标签列表，长度为 N，每个元素维度 (n_objects,)
            
        Returns:
            total_loss: 总损失（标量）
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)  # 441
        n_classes = predicted_scores.size(2)
        
        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)
        
        # 初始化真实标签张量
        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)
        
        # 为每张图像匹配先验框与真实目标
        for i in range(batch_size):
            n_objects = boxes[i].size(0)
            
            # 计算真实目标与所有先验框的 IoU（重叠度）
            overlap = find_jaccard_overlap(boxes[i], self.priors_xy)  # (n_objects, 441)
            
            # ==================== 先验框匹配策略 ====================
            # 为每个先验框找到 IoU 最大的真实目标
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (441,)
            
            # 问题：仅按上述方式匹配可能导致某些真实目标没有对应的正样本先验框
            # 原因1：某个目标可能不是任何先验框的"最佳匹配"
            # 原因2：即使是最佳匹配，IoU 也可能低于阈值（如 0.5）
            
            # 解决方案：反向匹配 - 为每个真实目标找到 IoU 最大的先验框
            _, prior_for_each_object = overlap.max(dim=1)  # (n_objects,)
            
            # 强制将这些先验框分配给对应的真实目标
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)
            
            # 人工提高这些先验框的 IoU 为 1.0，确保它们被视为正样本
            overlap_for_each_prior[prior_for_each_object] = 1.0
            
            # 为每个先验框分配类别标签
            label_for_each_prior = labels[i][object_for_each_prior]  # (441,)
            
            # 将 IoU 低于阈值的先验框标记为背景类（类别 0）
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0
            
            # 保存类别标签
            true_classes[i] = label_for_each_prior
            
            # 将真实边界框编码为相对于先验框的偏移量（与预测格式一致）
            # xy格式 → cxcy格式 → gcxgcy偏移格式
            true_locs[i] = cxcy_to_gcxgcy(
                xy_to_cxcy(boxes[i][object_for_each_prior]), 
                self.priors_cxcy
            )  # (441, 4)
        
        # ==================== 计算定位损失 ====================
        # 识别正样本先验框（包含目标，非背景）
        positive_priors = true_classes != 0  # (N, 441)
        
        # 仅对正样本计算定位损失（背景不需要定位）
        loc_loss = self.smooth_l1(
            predicted_locs[positive_priors],  # 预测的正样本位置
            true_locs[positive_priors]  # 真实的正样本位置
        )
        
        # ==================== 计算置信度损失 ====================
        # 采用难负样本挖掘（Hard Negative Mining）策略：
        # 1. 对所有正样本计算损失
        # 2. 对负样本，只选择损失最大的部分（最难分类的负样本）
        # 3. 负样本数量 = neg_pos_ratio × 正样本数量
        
        # 统计每张图像的正样本数量
        n_positives = positive_priors.sum(dim=1)  # (N,)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N,)
        
        # 计算所有先验框的分类损失
        conf_loss_all = self.cross_entropy(
            predicted_scores.view(-1, n_classes),  # (N*441, n_classes)
            true_classes.view(-1)  # (N*441,)
        )
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 441)
        
        # 提取正样本的损失
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives),)
        
        # ==================== 难负样本挖掘 ====================
        # 从负样本中选择损失最大的样本
        conf_loss_neg = conf_loss_all.clone()
        
        # 将正样本位置的损失置为 0（排除正样本）
        conf_loss_neg[positive_priors] = 0.0
        
        # 按损失从大到小排序
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 441)
        
        # 创建难度排名索引 [0, 1, 2, ..., 440]
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)
        
        # 选择前 n_hard_negatives 个最难的负样本
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 441)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives),)
        
        # 总置信度损失 = (难负样本损失 + 正样本损失) / 正样本总数
        # 注意：虽然包含了负样本，但仍然除以正样本数量（遵循 SSD 论文）
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()
        
        # ==================== 返回总损失 ====================
        # 总损失 = 置信度损失 + alpha × 定位损失
        return conf_loss + self.alpha * loc_loss