# import config
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
import numpy as np
# import os
# import random
# import torch

def batch_iou_all(boxes, box):
    """Compute the Intersection-Over-Union of a batch of boxes with another
    box.
    Args:
    box1: 2D array of [cx, cy, width, height].
    box2: a single array of [cx, cy, width, height]
    Returns:
    ious: array of a float number in range [0, 1].
    """
    lr = np.maximum(
        np.minimum(boxes[:,0]+0.5*boxes[:,2], box[0]+0.5*box[2]) - \
        np.maximum(boxes[:,0]-0.5*boxes[:,2], box[0]-0.5*box[2]),
        0
    )
    tb = np.maximum(
        np.minimum(boxes[:,1]+0.5*boxes[:,3], box[1]+0.5*box[3]) - \
        np.maximum(boxes[:,1]-0.5*boxes[:,3], box[1]-0.5*box[3]),
        0
    )
    inter = lr*tb
    union = boxes[:,2]*boxes[:,3] + box[2]*box[3] - inter  # iou case
    iou_max = box[2]*box[3] # anchors include bbox
    iou_max1 = boxes[:,2]*boxes[:,3]  # bbox include anchors
    return np.maximum(inter/union, inter/iou_max, inter/iou_max1)


# def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
#     """
#     Video explanation of this function:
#     https://youtu.be/XXYG5ZWtjj0
#     This function calculates intersection over union (iou) given pred boxes
#     and target boxes.
#     Parameters:
#         boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
#         boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
#         box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
#     Returns:
#         tensor: Intersection over union for all examples
#     """

#     if box_format == "midpoint":
#         box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
#         box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
#         box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
#         box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
#         box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
#         box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
#         box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
#         box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

#     if box_format == "corners":
#         box1_x1 = boxes_preds[..., 0:1]
#         box1_y1 = boxes_preds[..., 1:2]
#         box1_x2 = boxes_preds[..., 2:3]
#         box1_y2 = boxes_preds[..., 3:4]
#         box2_x1 = boxes_labels[..., 0:1]
#         box2_y1 = boxes_labels[..., 1:2]
#         box2_x2 = boxes_labels[..., 2:3]
#         box2_y2 = boxes_labels[..., 3:4]

#     x1 = torch.max(box1_x1, box2_x1)
#     y1 = torch.max(box1_y1, box2_y1)
#     x2 = torch.min(box1_x2, box2_x2)
#     y2 = torch.min(box1_y2, box2_y2)

#     intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
#     box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
#     box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

#     return intersection / (box1_area + box2_area - intersection + 1e-6)