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
def one_hot_label (labels, num_classes=2, smoothing=0.03, device='cuda') :
#     labels = torch.tensor([1, 1, 1, 0, -1])
    batch_size, seq_len = labels.shape
    row, col = torch.where(labels == -1.)
    
    labels = labels.reshape(batch_size, seq_len, 1)

    num_classes = num_classes
    one_hot_target = (labels == torch.arange(num_classes).reshape(1, num_classes).to(device)).float()
#     label smoothing
#     one_hot_target = torch.where(one_hot_target == 1., torch.ones(one_hot_target.shape).to(device) * (1. - smoothing), 
#                                  torch.ones(one_hot_target.shape).to(device) * (smoothing/(num_classes-1)))
#     one_hot_target[row,col,:] = 0. # inplace operations
    
    return one_hot_target


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
#         self.focal_loss = FocalLoss()
        self.mini = torch.tensor(1e-7)
 
    def forward(self, predictions, labels):
        # Check where obj and noobj (we ignore if labels == -1)
        cell_pred = torch.sigmoid(predictions[:,:,1])
#         cell_pred_clone = cell_pred.clone()
        ab_cell_pred = torch.nn.Softmax(dim=2)(predictions[:,:,1:])
#         cell_pred = predictions[:,:,0]
#         ab_cell_pred = predictions[:,:,1]
        
        cell_label = labels[:,:,1]
        
#         ab_cell_label = labels[:,:,1]
#         num_ab_cell = (ab_cell_label > 0.9).sum()
#         ab_cell_label = torch.where(torch.ne(cell_label, 1.), (torch.ones(ab_cell_label.shape) * (-1)).to(self.device),
#                                    ab_cell_label)
        
#         one_hot_target = one_hot_label(ab_cell_label)

        batch_size, _ = cell_pred.shape
        
        ab_loss = []
        multi_class_loss = []
        batch_loss = []
        for i in range(batch_size) :
            focal_weight = torch.where(torch.eq(cell_label[i], 1.), 1. - cell_pred[i], cell_pred[i])
            focal_weight = torch.pow(focal_weight, self.gamma)            
            
            cell_loss = -(cell_label[i] * torch.log(cell_pred[i] + self.mini) + (1. - cell_label[i]) * torch.log((1. - cell_pred[i]) + self.mini))
            cell_loss *= focal_weight
            cell_loss = torch.where(torch.ne(cell_label[i], -1.0), cell_loss, torch.zeros(cell_loss.shape).to(self.device))
#             cell_loss = torch.where(torch.ne(cell_label[i], -1.), cell_loss, torch.zeros(cell_loss.shape).to(self.device))           
            
# #             num_cell = (cell_label[i] > 0.9).sum()
# #             num_ab = (ab_cell_label[i] > 0.9).sum()
            
#             alpha = torch.zeros(one_hot_target[i].shape)
#             alpha[:,0] = 0.25 #(num_ab)/(num_cell+num_ab+1)
#             alpha[:,1] = 0.75 #(num_cell)/(num_cell+num_ab+1)
            
#             pt = torch.where(one_hot_target[i] > 0.9, ab_cell_pred[i], 1-ab_cell_pred[i])
#             ce_loss = -(one_hot_target[i] * torch.log(ab_cell_pred[i] + self.mini))
#             ab_loss = (ce_loss * ((1-pt) ** self.gamma)) * alpha.to(self.device)
            
#             multi_class_loss.append(ce_loss.sum())
            
#             batch_loss.append(self.abnormal * ab_loss.mean(dim=-1).mean() + cell_loss.mean() )
            batch_loss.append(cell_loss.mean())
        
        return torch.stack(batch_loss).mean(), torch.tensor(0), torch.tensor(0)
        
        
        
        
# #         this is for removing backgroud cell
# #         cell_check = torch.where(torch.ge(cell_pred, self.cell_threshold), torch.zeros(cell_label.shape).to(self.device), 
# #                                  (torch.ones(cell_label.shape) * (-1)).to(self.device))
# #         cell_check = torch.where(torch.ne(cell_label, 0.), (torch.ones(cell_label.shape) * (-1)).to(self.device), 
# #                                  torch.zeros(cell_label.shape).to(self.device))
        
# # #         binary cross entropy for normal cell
#         bce_loss = -(cell_label * torch.log(cell_pred + self.mini) + (1. - cell_label) * torch.log((1. - cell_pred) + self.mini))
#         bce_loss = torch.where(torch.ne(cell_label, -1.), bce_loss, torch.zeros(bce_loss.shape).to(self.device))

#         one_hot_target = one_hot_label(ab_cell_label)

# # #         focal loss for multi class, alpha is not adjusted yet
# #         pt = torch.where(one_hot_target > 0.9, ab_cell_pred, 1-ab_cell_pred)
# #         multi_class_loss = -(one_hot_target * torch.log(ab_cell_pred + self.mini)) * ((1-pt) ** self.gamma)

#         print(ab_cell_pred.shape)
#         multi_class_loss = self.focal_loss(ab_cell_pred, one_hot_target)
#         denom = (multi_class_loss > 0).sum()
        
# #         removing the background cell
# #         multi_class_loss = torch.where(torch.ne(torch.stack([cell_check, cell_check], dim=-1),-1.), 
# #                                        multi_class_loss, torch.zeros(multi_class_loss.shape).to(self.device))
    
# # #     removing target cell neighbor area
# #         multi_class_loss = torch.where(torch.ne(torch.stack([ab_cell_label, ab_cell_label], dim=-1),-1.), 
# #                                        multi_class_loss, torch.zeros(multi_class_loss.shape).to(self.device))   
#         batch_loss = []
#         for i in range(batch_size) :
# #             batch_loss.append(bce_loss.mean(dim=-1) + self.abnormal * multi_class_loss.mean(dim=-1).mean(dim=-1))
#             batch_loss.append(self.abnormal * multi_class_loss.mean(dim=-1).mean(dim=-1))
            
#         return torch.stack(batch_loss).mean(), self.abnormal * multi_class_loss.mean()/(denom+1), bce_loss.mean(), num_ab_cell, denom

# #         return  bce_loss.mean(dim=-1).mean() + self.abnormal * multi_class_loss.mean(dim=-1).mean(dim=-1).mean(), \
# #                 self.abnormal * multi_class_loss.sum(), bce_loss.mean(), num_ab_cell, num_ab_cell1
                     
