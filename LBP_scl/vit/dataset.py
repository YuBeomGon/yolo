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
from config import *

from utils import batch_iou_all

ImageFile.LOAD_TRUNCATED_IMAGES = True


train_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomResizedCrop(height=IMAGE_SIZE,width=IMAGE_SIZE,scale=[0.9,1.0],ratio=[0.9,1.1],p=0.5),
], p=1.0, bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0.8))    

val_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1),
], p=1.0, bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0.8))    

test_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1),
], p=1.0)

# def get_data(img_id):
#     if img_id not in df_data.groups:
#         return dict(image_id=img_id, boxes=list())
    
#     data  = df_data.get_group(img_id)
#     boxes = data[['xmin', 'ymin', 'w', 'h', 'label_id']].values
#     return dict(image_id = img_id, boxes = boxes)

class LbpDataset(Dataset):
    def __init__(
        self,
        image_list,
        transform=None,
        stride=96,
        kernel_size=128,
    ):
        self.image_list = image_list
        self.transform = transform
        self.default_path = '/home/Dataset/scl/'
        self.stride = stride
        self.kernel_size = kernel_size
        self.threshold = 200
        self.kernel = torch.ones((1, 1, self.kernel_size, self.kernel_size))
        self.image_size = IMAGE_SIZE
        self.grid_size = 1 + int((self.image_size - self.kernel_size) / self.stride )
               

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        path = self.image_list[index]['image_id']
#         bbox is coco dataformat
#         xmin, ymin, width, height
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
#         print(image.shape)        

        cell_iou = F.conv2d(torch.tensor(mask).reshape(1,1,self.image_size,self.image_size).float(), 
                            self.kernel, stride=self.stride)
        cell_iou /= (self.kernel_size * self.kernel_size)
        cell_iou = torch.squeeze(cell_iou).view(self.grid_size*self.grid_size, -1)    

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale     
        if len(boxes) > 0 :
            for box in boxes :
                x, y, width, height, class_label = map(int, box)
                targets_mask = torch.zeros(1,1,self.image_size,self.image_size)
                targets_mask[:,:,y:y+height,x:x+width] = 1
            targets_mask = targets_mask * mask
            targets = F.conv2d(targets_mask.float(), self.kernel, stride=self.stride) / (self.kernel_size * self.kernel_size)

            for box in boxes :
                x, y, width, height, class_label = box
                i, j = int(self.grid_size * (y+height/2)/self.image_size), int(self.grid_size * (x+width/2)/self.image_size)
                targets[0,0,i,j] = 1.                       

            targets = torch.squeeze(targets).view(self.grid_size*self.grid_size, -1)
            
        else :
            boxes = [[0, 0, 1, 1, 1.0]]
            targets = torch.zeros(self.grid_size * self.grid_size, 1)
        
        return image, cell_iou, targets, path

def get_hard_label(cell_iou, targets) :
    cell_iou[(cell_iou < 0.1)] = 0.0
    cell_iou[(cell_iou >= 0.1) & (cell_iou < 0.7)] = -1
    cell_iou[(cell_iou >= 0.7)] = 1.   

    targets[(targets == 0.)] = 0.0
    targets[(targets > 0.) & (targets < 0.7)] = -1.
    targets[(targets >= 0.7) & (targets < 0.9)] = 1.
    targets[(targets >= 0.9)] = 1.   
    
    labels = torch.zeros(cell_iou.shape)
    labels = torch.where(torch.eq(cell_iou, 1.), torch.ones(cell_iou.shape), labels)
    labels = torch.where(torch.eq(targets, 1.), torch.ones(targets.shape)*(2), labels)
    
    return labels

def get_indices (cell_iou, targets) :
    normal_cell = torch.where(cell_iou[0,:,0] >= 0.85)[0]
    normal_cell_not = torch.where((cell_iou[0,:,0] > 0.0) & (cell_iou[0,:,0] < 0.85))[0]

    abnormal_cell = torch.where(targets[0,:,0] >= 0.5)[0]
    abnormal_cell_not = torch.where((targets[0,:,0] <= 0.0))[0]

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
