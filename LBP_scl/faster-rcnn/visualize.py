import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

def visualize_bbox(img, boxes, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    bbox = boxes[:4]
    if len(boxes) ==5 and boxes[4] ==1 :
        color = (0, 255, 0)
    x_min, y_min, x_max, y_max = list(map(int, bbox))
#     print(bbox)
#     x_min, y_min, x_max, y_max = list(map(round, bbox))
#     print((int(x_min), int(y_min)), (int(x_max), int(y_max)))

    img = cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=color, thickness=thickness)
    return img

def visualize(image, bboxes):
    img = image.copy()
    print(img.shape)
#     img = image.clone().detach()
    for bbox in (bboxes):
#         print(bbox)
        img = visualize_bbox(img, bbox)
    plt.figure(figsize=(7, 7))
    plt.axis('off')
    plt.imshow(img)
