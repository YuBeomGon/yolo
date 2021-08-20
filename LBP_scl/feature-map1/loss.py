"""
Implementation of Yolo Loss Function similar to the one in Yolov3 paper,
the difference from what I can tell is I use CrossEntropy for the classes
instead of BinaryCrossEntropy.
"""
import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

# # one hot encoding and label smoothing
# def one_hot (labels, num_classes=2, smoothing=0.03, device='cuda') :
# #     labels = torch.tensor([1, 1, 1, 0, -1])
# #     batch_size, seq_len = labels.shape    
# #     row, col = torch.where(labels == -1.)
# #     labels = labels.reshape(batch_size, seq_len, 1)
#     seq_len = labels.shape
#     labels = labels.reshape(seq_len, 1)

#     num_classes = num_classes
#     one_hot_target = (labels == torch.arange(num_classes).reshape(1, num_classes).to(device)).float()
# #     label smoothing
# #     one_hot_target = torch.where(one_hot_target == 1., torch.ones(one_hot_target.shape).to(device) * (1. - smoothing), 
# #                                  torch.ones(one_hot_target.shape).to(device) * (smoothing/(num_classes-1)))
# #     one_hot_target[row,col,:] = 0. # inplace operations
    
#     return one_hot_target

class Clamp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0.0003, max=0.9997)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


def one_hot(index, classes=2):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0).cuda()
    index = index.view(*view)
    ones = 1.

    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1)).cuda()
        mask = Variable(mask, volatile=index.volatile).cuda()

    return mask.scatter_(1, index, ones)

class FocalLoss(nn.Module):

    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, logit, targets):
        targets = targets.clamp(0, 1).long()
        y = one_hot(targets).cuda()
#         logit = F.softmax(input, dim=-1)
        print(logit.shape)
        print(targets.shape)
        print(y.shape)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss

        return loss.sum()

class LBPloss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        self.device = device
        self.alpha = 0.25
        self.gamma = 2.0
        self.abnormal = 5.0
        self.cell_threshold = 0.7
        self.focal_loss = FocalLoss()
        self.mini = torch.tensor(3e-4)
        self.clamp = Clamp().apply
 
    def forward(self, cell_out, ab_cell_out, multi_out, point, labels):
        # Check where obj and noobj (we ignore if labels == -1)
        batch_size, _, _ = cell_out.shape
        
        cell_pred = cell_out[:,:,0]
        cell_pred = self.clamp(cell_pred)
        cell_label = labels[:,:,0]
        batch_loss = []
        for i in range(batch_size) :
            label_sum = (cell_label[i] > -1.).sum()
            alpha_factor = torch.ones(cell_label[i].shape).to(self.device) * self.alpha
            alpha_factor = torch.where(torch.eq(cell_label[i], 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(cell_label[i], 1.), 1. - cell_pred[i], cell_pred[i])
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)            
            
            cell_loss = -(cell_label[i] * torch.log(cell_pred[i]) + (1. - cell_label[i]) * torch.log((1. - cell_pred[i])))
            cell_loss *= focal_weight
            cell_loss = torch.where(torch.ne(cell_label[i], -1.0), cell_loss, torch.zeros(cell_loss.shape).to(self.device))
            
            batch_loss.append(cell_loss.sum()/label_sum)
            
        ab_cell_pred = ab_cell_out[:,:,0]
        ab_cell_pred = self.clamp(ab_cell_pred)
        ab_cell_label = labels[:,:,1]
        ab_batch_loss = []
        for i in range(batch_size) :
            ab_label_sum = (ab_cell_label[i] > -1.).sum()
            alpha_factor = torch.ones(ab_cell_label[i].shape).to(self.device) * self.alpha
            alpha_factor = torch.where(torch.eq(ab_cell_label[i], 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(ab_cell_label[i], 1.), 1. - ab_cell_pred[i], ab_cell_pred[i])
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)            
            
            ab_cell_loss = -(ab_cell_label[i] * torch.log(ab_cell_pred[i]) + (1. - ab_cell_label[i]) * torch.log((1. - ab_cell_pred[i])))
            ab_cell_loss *= focal_weight
            ab_cell_loss = torch.where(torch.ne(ab_cell_label[i], -1.0), ab_cell_loss, torch.zeros(ab_cell_loss.shape).to(self.device))
            
            ab_batch_loss.append(ab_cell_loss.sum()/ab_label_sum)  
            
# center point bbox regression            
        ab_point_pred = point[:,:,:]
#         ab_cell_pred = self.clamp(ab_cell_pred)
        ab_point_label = labels[:,:,1:4]
        ab_point_loss = []
        for i in range(batch_size) :
#             ab_point_sum = (ab_point_label[i][:,0] == 1.).sum()
            row = torch.where(ab_point_label[i,:,0] == 1.)
            point_loss = []
            if len(row[0]) > 0 :
                for j in row :
#                     if not torch.isfinite(self.mse(ab_point_pred[i,j,0:2], ab_point_label[i,j,1:3])) :
                    point_loss.append(self.mse(ab_point_pred[i,j,0:2], ab_point_label[i,j,1:3]))
            else :
                point_loss.append(torch.tensor(0.).to(self.device))
            ab_point_loss.append(torch.stack(point_loss).mean())              

        multi_pred = multi_out[:,:,0]
        multi_pred = self.clamp(multi_pred)
        multi_label = labels[:,:,4]            
        multi_loss = []
        for i in range(batch_size) :
            multi_label_sum = (multi_label[i] > -1.).sum()
            alpha_factor = torch.ones(multi_label[i].shape).to(self.device) * self.alpha
            alpha_factor = torch.where(torch.eq(multi_label[i], 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(multi_label[i], 1.), 1. - multi_pred[i], multi_pred[i])
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)            
            
            multi_cell_loss = -(multi_label[i] * torch.log(multi_pred[i]) + (1. - multi_label[i]) * torch.log((1. - multi_pred[i])))
            multi_cell_loss *= focal_weight
            multi_cell_loss = torch.where(torch.ne(multi_label[i], -1.0), multi_cell_loss, torch.zeros(multi_cell_loss.shape).to(self.device))
            
            multi_loss.append(multi_cell_loss.sum()/multi_label_sum)                
        
        return 0.5*torch.stack(batch_loss).mean() + 5.*torch.stack(ab_batch_loss).mean() + \
    5.*torch.stack(multi_loss).mean() + 1.*torch.stack(ab_point_loss).mean(), torch.stack(batch_loss).mean(), torch.stack(ab_batch_loss).mean(), torch.stack(multi_loss).mean(), torch.stack(ab_point_loss).mean()
                     
# class FocalLoss(nn.modules.loss._WeightedLoss):
#     def __init__(self, weight=None, gamma=2,reduction='mean', device='cuda'):
#         super(FocalLoss, self).__init__(weight,reduction=reduction)
#         self.gamma = gamma
#         self.weight = weight #weight parameter will act as the alpha parameter to balance class weights
#         self.device = device

#     def forward(self, input, target):

#         print(input.shape)
#         print(target.shape)
#         print(input)
#         print(target)
#         ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
# #         ce_loss = -(target * torch.log(input[:,1] + 1e-8) + 
# #                              (1. - target) * torch.log((input[:,0]) + 1e-8 ))
#         print(ce_loss.shape)

#         pt = torch.exp(-ce_loss)
#         focal_loss = ((1 - pt) ** self.gamma * ce_loss)
# #         cell_check = torch.stack([cell_check, cell_check], dim=1)
# #         loss = torch.where(torch.ne(cell_check, -1.), focal_loss, torch.zeros(focal_loss.shape).to(self.device))
# #         ab_loss = torch.where(torch.ne(target, -1.), loss, torch.zeros(loss.shape).to(self.device))
        
# #         return ab_loss.sum()
#         return focal_loss
