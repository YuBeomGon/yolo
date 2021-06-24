"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import os
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import albumentations as A
import albumentations.pytorch

def switch_image(img) :
    h, w = img.shape[:2]
    if (h, w) == (4032, 1960) or (h, w) == (4000, 1800) :
        img = np.flip(img, 1)
        img = np.transpose(img, (1, 0, 2))      
    return img

train_transforms = A.Compose([
    A.CenterCrop(1200,1200, True,1),
    A.Resize(400, 400, p=1),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomResizedCrop(height=400,width=400,scale=[0.95,1.05],ratio=[0.95,1.05],p=0.5),
    A.pytorch.ToTensor(),
], p=1.0, bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0.8))

val_transforms = A.Compose([
    A.CenterCrop(1200,1200, True,1),
    A.Resize(400, 400, p=1),
    A.pytorch.ToTensor(),
], p=1.0, bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0.8))


class PapsDataset(torch.utils.data.Dataset):
    def __init__(
        self, labels, partition, S=25, B=2, C=1, transform=None,
    ):
        self.partition = partition
        self.labels = labels
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        self.default_path = '/home/Dataset/Papsmear/original/'

    def __len__(self):
        return len(self.partition)

    def __getitem__(self, index):
        path = self.partition[index]
        boxes = self.labels[path]

        image = cv2.imread(self.default_path + path)
        image = switch_image(image)
        
#         print(boxes)
        if self.transform:
            # image = self.transform(image)
            transformed_image = self.transform(image=image, bboxes=boxes)
            image = transformed_image['image']
            boxes = transformed_image['bboxes']
#         print(image.shape)
        if len(boxes) == 0 :
            boxes = [[0, 0, 1e-6, 1e-6, 1.0]]
        boxes = torch.tensor(boxes)
#         print(boxes)
        boxes[:,2] = (boxes[:,2] - boxes[:,0])/2
        boxes[:,3] = (boxes[:,3] - boxes[:,1])/2
        boxes[:,0] = boxes[:,0] + boxes[:,2]
        boxes[:,1] = boxes[:,1] + boxes[:,3]
        boxes[:,:4] = boxes[:,:4]/image.shape[1]
#         print(boxes)

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        for box in boxes:
            x, y, width, height, class_label = box.tolist()
#             class_label = int(class_label)
            class_label = 0

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )
#             print(i, j, x_cell, y_cell, width_cell, height_cell)

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, 1] == 0:
                # Set that there exists an object
                label_matrix[i, j, 1] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
#                 print('box_coordinates', box_coordinates)

                label_matrix[i, j, 2:6] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1
#                 print('label_matrix', label_matrix[i,j,:])
#             elif label_matrix[i, j, 6] == 0:
# #                 print('already one room is assigned')
#                 # Set that there exists an object
#                 label_matrix[i, j, 6] = 1

#                 # Box coordinates
#                 box_coordinates = torch.tensor(
#                     [x_cell, y_cell, width_cell, height_cell]
#                 )

#                 label_matrix[i, j, 7:11] = box_coordinates

#                 # Set one hot encoding for class_label
#                 label_matrix[i, j, class_label] = 1
#             elif label_matrix[i, j, 11] == 0:
# #                 print('already one room is assigned')
#                 # Set that there exists an object
#                 label_matrix[i, j, 11] = 1

#                 # Box coordinates
#                 box_coordinates = torch.tensor(
#                     [x_cell, y_cell, width_cell, height_cell]
#                 )

#                 label_matrix[i, j, 12:16] = box_coordinates

#                 # Set one hot encoding for class_label
#                 label_matrix[i, j, class_label] = 1                
            else :
#                 print('no more rooms for this label')
                pass

        return image, label_matrix, boxes
    
class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            # image = self.transform(image)
            image, boxes = self.transform(image, boxes)

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, 20] == 0:
                # Set that there exists an object
                label_matrix[i, j, 20] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return image, label_matrix    