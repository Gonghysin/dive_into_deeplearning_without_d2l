# 主要是目标检测中的一些工具函数

import torch
import torchvision

def xy_to_cxcy(xy):
    """将xy坐标转换为cxcy坐标"""
    return torch.cat([(xy[:, 2:] + xy[:, :2]) /2 , # c_x, c_y
                      (xy[:, 2:] - xy[:, :2])], 1) # w, h

def cxcy_to_xy(cxcy):
    """将cxcy坐标转换为xy坐标"""
    return torch.cat([cxcy[:, :2] - cxcy[:, 2:] / 2, # x_min, y_min
                      cxcy[:, :2] + cxcy[:, 2:] / 2], 1) # x_max, y_max

def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)                                                                                                           
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]

def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """
    
    # Find intersections
    intersection = find_intersection(set_1, set_2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:,2] - set_1[:, 0 ]) * (set_1[:,3] - set_1[:, 1 ])
    areas_set_2 = (set_2[:,2] - set_2[:, 0 ]) * (set_2[:,3] - set_2[:, 1 ])

    # Find the union
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection

    return intersection / union

