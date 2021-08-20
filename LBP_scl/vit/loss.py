"""
Implementation of Yolo Loss Function similar to the one in Yolov3 paper,
the difference from what I can tell is I use CrossEntropy for the classes
instead of BinaryCrossEntropy.
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# one hot encoding and label smoothing
def one_hot_label (labels, num_classes=3, smoothing=0.03, device='cuda') :
#     labels = torch.tensor([1, 1, 1, 0, -1])
    batch_size, seq_len, c = labels.shape
#     row, col = torch.where(labels == -1.)
    
    labels = labels.reshape(batch_size, seq_len, -1)

    num_classes = num_classes
    one_hot_target = (labels == torch.arange(num_classes).reshape(1, num_classes).to(device)).float()
#     label smoothing
#     one_hot_target = torch.where(one_hot_target == 1., torch.ones(one_hot_target.shape).to(device) * (1. - smoothing), 
#                                  torch.ones(one_hot_target.shape).to(device) * (smoothing/(num_classes-1)))
#     one_hot_target[row,col,:] = 0. # inplace operations
    
    return one_hot_target


class LBPloss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.alpha = 0.25
        self.gamma = 2.0
        self.abnormal = 5.0
        self.cell_threshold = 0.7
#         self.focal_loss = FocalLoss()
        self.mini = torch.tensor(1e-7)
 
    def forward(self, predictions, labels):
        # Check where obj and noobj (we ignore if labels == -1)
        cell_pred = torch.nn.Softmax(dim=2)(predictions[:,:,:])
        cell_pred = torch.clamp(cell_pred, min=self.mini, max=1-self.mini)
        cell_label = labels[:,:,]
        
        num_ab_cell = (cell_label > 1.0).sum()
        
        one_hot_target = one_hot_label(cell_label)

        batch_size, _, _ = cell_pred.shape
        
        multi_class_loss = []
        batch_loss = []
        for i in range(batch_size) :
            alpha = torch.ones(one_hot_target[i].shape) * (0.1)
            alpha[:,2] = 0.8
            alpha = alpha.to(self.device)
#             alpha = torch.where(torch.eq)
            
            pt = torch.where(one_hot_target[i] > 0.9, cell_pred[i], 1-cell_pred[i])
            ce_loss = -(one_hot_target[i] * torch.log(cell_pred[i]))
            ab_loss = (ce_loss * ((1-pt) ** self.gamma)) * alpha
            
            multi_class_loss.append(ce_loss.sum())
            
            batch_loss.append(ab_loss.sum(dim=-1).mean())
        
        return torch.stack(batch_loss).mean(), torch.stack(multi_class_loss).mean(), num_ab_cell