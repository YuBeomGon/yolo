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
    A.RandomResizedCrop(height=IMAGE_SIZE,width=IMAGE_SIZE,scale=[0.9,1.0],ratio=[0.9,1.1],p=0.5),
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

def gauss2D(shape=(32,32),sigma=4.):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

gaussian = gauss2D()

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
        self.resize_ratio = 4
        self.image_size = IMAGE_SIZE
        self.kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size)
        
        self.res_stride = int(stride/self.resize_ratio)
        self.res_kernel_size = int(kernel_size/4)
        self.res_img_size = int(self.image_size/self.resize_ratio)
        self.res_kernel = torch.ones(1, 1, self.res_kernel_size, self.res_kernel_size)
        
        self.grid_size = 1 + int((self.image_size - self.kernel_size) / self.stride )
        self.gaussian = torch.tensor(gauss2D(shape=(self.res_kernel_size, self.res_kernel_size))).view(1,1,self.res_kernel_size, self.res_kernel_size).float()
        

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
        resized_gray = cv2.resize(gray, (self.res_img_size, self.res_img_size))
#         gray = cv2.medianBlur(gray,5)

        image = (image.astype(np.float32))/255

#         image = (image.astype(np.float32)-127.5)/127.5

        laplacian = cv2.Laplacian(resized_gray,cv2.CV_8U,ksize=3)
        laplacian_iou = F.conv2d(torch.tensor(laplacian).reshape(1,1,self.res_img_size,self.res_img_size).float(), 
                    self.gaussian, stride=self.res_stride)
        laplacian_iou /= self.res_kernel_size ** 2
#         print(laplacian_iou.shape)
        laplacian_iou = torch.squeeze(laplacian_iou).view(self.grid_size*self.grid_size, -1) 

# mask using global mask
        mask = resized_gray < self.threshold        
        cell_iou = F.conv2d(torch.tensor(mask).reshape(1,1,self.res_img_size,self.res_img_size).float(), 
                            self.res_kernel, stride=self.res_stride)
        cell_iou /= self.res_kernel_size ** 2
        cell_iou = torch.squeeze(cell_iou).view(self.grid_size*self.grid_size, -1)    
        
        cell_iou = laplacian_iou * cell_iou
        cell_iou /= torch.max(cell_iou)

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale     
        if len(boxes) > 0 :
            mask = gray < self.threshold
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
#             print('No annotation')
            boxes = [[0, 0, 1, 1, 1.0]]
            targets = torch.zeros(self.grid_size * self.grid_size, 1)   

#         return image, cell_iou, targets, path, boxes
        return image, cell_iou, targets, path

# make hard label
def get_hard_label(cell_iou, targets) :
    cell_iou[(cell_iou < 0.1)] = 0.0
    cell_iou[(cell_iou >= 0.1) & (cell_iou < 0.3)] = -1
    cell_iou[(cell_iou >= 0.3)] = 1.   

    targets[(targets < 0.1)] = 0.0
    targets[(targets >= 0.1) & (targets < 0.7)] = -1.
    targets[(targets >= 0.7) & (targets < 0.9)] = 1.
    targets[(targets >= 0.9)] = 1.        
    
    return cell_iou, targets

# for sampling to make class balance
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
