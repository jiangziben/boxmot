import numpy as np
from scipy.optimize import linear_sum_assignment

def compute_iou_matrix(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    
    # 计算每个框的面积
    area1 = (boxes1[:, 2] - boxes1[:, 0] + 1) * (boxes1[:, 3] - boxes1[:, 1] + 1)
    area2 = (boxes2[:, 2] - boxes2[:, 0] + 1) * (boxes2[:, 3] - boxes2[:, 1] + 1)
    
    # 找到交集的坐标
    xA = np.maximum(boxes1[:, None, 0], boxes2[:, 0])
    yA = np.maximum(boxes1[:, None, 1], boxes2[:, 1])
    xB = np.minimum(boxes1[:, None, 2], boxes2[:, 2])
    yB = np.minimum(boxes1[:, None, 3], boxes2[:, 3])

    # 计算交集的面积
    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
    
    # 计算IOU矩阵
    iou_matrix = interArea / (area1[:, None] + area2 - interArea)
    
    return iou_matrix

# 示例 bbox 集合
boxes1 = [[10, 20, 30, 40], [15, 25, 35, 45]]
boxes2 = [[18, 28, 33, 43],[11, 22, 29, 39]]

# 计算IOU矩阵
iou_matrix = compute_iou_matrix(boxes1, boxes2)

# 将IOU矩阵转换为成本矩阵（1 - IOU，IOU越大成本越小）
cost_matrix = 1 - iou_matrix

# 使用匈牙利算法进行匹配
row_ind, col_ind = linear_sum_assignment(cost_matrix)
matches = list(zip(row_ind, col_ind))

print("匹配结果:", matches)
