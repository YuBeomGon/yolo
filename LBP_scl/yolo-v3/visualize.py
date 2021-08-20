import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

def visualize_bbox(img, bbox, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
#     x_min, y_min, x_max, y_max = list(map(int, bbox))
#     print(bbox)
    bbox_color = [int(c) for c in color]
    x_center, y_center, w, h = (bbox)
    x_min = x_center - w
    y_min = y_center - h
    x_max = x_center + w
    y_max = y_center + h
#     x_min, y_min, x_max, y_max = list(map(round, bbox))
#     print((int(x_min), int(y_min)), (int(x_max), int(y_max)))

    img = cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=bbox_color, thickness=thickness)
    return img

def visualize(image, bboxes):
    img = image.copy()
    print(img.shape)
#     img = image.clone().detach()
    for bbox in (bboxes):
#         print(bbox)
        img = visualize_bbox(img, bbox)
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(img)