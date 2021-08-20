"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""

# import config
import numpy as np
import os
import pandas as pd
import torch
import albumentations as A
import albumentations.pytorch
import cv2

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import batch_iou_all

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_SIZE = 2048

train_transforms = A.Compose([
#     A.CenterCrop(1248,1248, True,1),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomResizedCrop(height=IMAGE_SIZE,width=IMAGE_SIZE,scale=[0.8,1.0],ratio=[0.8,1.25],p=0.5),
#     A.pytorch.ToTensor(),
], p=1.0, bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0.8))    
# ], p=1.0, bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0.8, label_fields=['labels']))

val_transforms = A.Compose([
#     A.CenterCrop(1248,1248, True,1),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1),
#     A.pytorch.ToTensor(),
], p=1.0, bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0.8))    
# ], p=1.0, bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0.8, label_fields=['labels']))

test_transforms = A.Compose([
#     A.CenterCrop(1248,1248, True,1),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1),
#     A.pytorch.ToTensor(),
], p=1.0)

class LbpDataset(Dataset):
    def __init__(
        self,
        image_list,
        transform=None,
        stride=32,
        kernel_size=128,
    ):
        self.image_list = image_list
        self.transform = transform
        self.default_path = '/home/Dataset/scl/'
        self.stride = stride
        self.kernel_size = kernel_size
        self.threshold = 200
        self.kernel = torch.ones((1, 1, kernel_size, kernel_size))
        self.image_size = IMAGE_SIZE
        self.grid_size = 1 + int((self.image_size - self.kernel_size) / self.stride )
#         self.anchors = torch.zeros([self.grid_size, self.grid_size,4])
        self.anchors = np.zeros([self.grid_size, self.grid_size,4])
        
        for x in range(self.grid_size) :          
            for y in range(self.grid_size) :
                self.anchors[x][y][1] = self.stride * x
                self.anchors[x][y][3] = self.stride * x + self.kernel_size   
                self.anchors[x][y][0] = self.stride * y
                self.anchors[x][y][2] = self.stride * y + self.kernel_size
                
#                 self.anchors[x][y][0] = self.stride * x
#                 self.anchors[x][y][2] = self.stride * x + self.kernel_size   
#                 self.anchors[x][y][1] = self.stride * y
#                 self.anchors[x][y][3] = self.stride * y + self.kernel_size                

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        path = self.image_list[index]['image_id']
        boxes = self.image_list[index]['boxes']

        image = cv2.imread(self.default_path + path)

        if self.transform:
            augmentations = self.transform(image=image, bboxes=boxes)
#             augmentations = self.transform(image=image, bboxes=boxes, labels=labels)
            image = augmentations["image"]
            boxes = augmentations["bboxes"]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray,5)
        mask = gray < self.threshold
        image = (image.astype(np.float32)-127.5)/127.5

        cell_iou = F.conv2d(torch.tensor(mask).reshape(1,1,self.image_size,self.image_size).float(), self.kernel, stride=self.stride)
        cell_iou /= (self.kernel_size * self.kernel_size)
        cell_iou = torch.squeeze(cell_iou).view(self.grid_size*self.grid_size, -1)    

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = torch.ones(self.grid_size * self.grid_size) * (-1.)
        if len(boxes) > 0 :
            boxes = np.array(boxes) 
            boxes[:,2] = boxes[:,0] + boxes[:,2]
            boxes[:,3] = boxes[:,1] + boxes[:,3]
            
            iou_list = []
            for box in boxes :
                iou = batch_iou_all(self.anchors.reshape(-1,4), box[:4])
                iou_list.append(iou)
            iou_max = np.max(iou_list, axis=0)
            iou_max = iou_max * torch.squeeze(cell_iou).numpy()

            targets[iou_max < 0.3] = 0.05
            targets[(iou_max >= 0.8) & (iou < 0.95)] = 0.9
            targets[(iou_max >= 0.95)] = 0.95
        
#         if len(boxes) > 0 :
#             for box in boxes :
#                 x, y, width, height, class_label = box
#                 i, j = int(self.grid_size * y), int(self.grid_size * x)
#                 targets[i,j] = 1.
        else :
#             print('No annotation')
            boxes = [[0, 0, 1, 1, 1.0]]
    
        targets = targets.view(self.grid_size * self.grid_size, -1)
        
        cell_iou[(cell_iou < 0.2)] = 0.03
        cell_iou[(cell_iou >= 0.2) & (cell_iou < 0.8)] = -1
        cell_iou[(cell_iou >= 0.8) & (cell_iou < 0.9)] = 0.85    
        cell_iou[(cell_iou >= 0.9)] = 0.97        

        return image, cell_iou, targets, path

def get_indices (cell_iou, targets) :
    normal_cell = torch.where(cell_iou[0,:,0] >= 0.85)[0]
    normal_cell_not = torch.where((cell_iou[0,:,0] > 0.0) & (cell_iou[0,:,0] < 0.85))[0]

    abnormal_cell = torch.where(targets[0,:,0] >= 0.85)[0]
    abnormal_cell_not = torch.where((targets[0,:,0] > 0.0) & (targets[0,:,0] < 0.85))[0]

    normal_cell_indices = torch.randperm(len(normal_cell))[:5]
    normal_cell_not_indices = torch.randperm(len(normal_cell_not))[:5]
    abnormal_cell_indices = torch.randperm(len(abnormal_cell))[:5]
    abnormal_cell_not_indices = torch.randperm(len(abnormal_cell_not))[:5]   

    ncell = (normal_cell[normal_cell_indices])
    ncell_not = (normal_cell_not[normal_cell_not_indices])
    abcell = (abnormal_cell[abnormal_cell_indices])
    abcell_not = (abnormal_cell_not[abnormal_cell_not_indices])  

    indices, _ = torch.sort(torch.cat([ncell, ncell_not, 
               abcell, abcell_not]), dim=-1)    
    
    return indices, cell_iou[:,indices, :], targets[:,indices,:]
