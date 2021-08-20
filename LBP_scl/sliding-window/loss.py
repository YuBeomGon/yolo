"""
Implementation of Yolo Loss Function similar to the one in Yolov3 paper,
the difference from what I can tell is I use CrossEntropy for the classes
instead of BinaryCrossEntropy.
"""
import random
import torch
import torch.nn as nn

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
 
    def forward(self, predictions, target):
        # Check where obj and noobj (we ignore if target == -1)
        
        cell_pred = predictions[:,:,0]
        ab_cell_pred = predictions[:,:,1]
        
        cell_label = target[:,:,0]
        ab_cell_label = target[:,:,1]

        batch_size, _ = cell_pred.shape
#         losses = []
        bce = []
        cell_loss = []
        bce_ab = []
        cell_loss_ab = []
        
#         cell_label[(cell_label >= 0.) & (cell_label < 0.1)] = 0
#         cell_label[(cell_label >= 0.1) & (cell_label < 0.7)] = -1
#         cell_label[(cell_label >= 0.7)] = 1   
        
        for j in range(batch_size) :

            # bce for normal cell
#             print(-(cell_label[j] * torch.log(cell_pred[j]) + (1. - cell_label[j]) * torch.log(1. - cell_pred[j]) ))
            bce.append(-(cell_label[j] * torch.log(cell_pred[j] + 1e-8) + (1. - cell_label[j]) * torch.log((1. - cell_pred[j]) + 1e-8)))
            cell_loss.append(torch.where(torch.ne(cell_label[j], -1.), bce[j], torch.zeros(bce[j].shape).to(self.device)))

            # bce for abnormal cell
            bce_ab.append( -(ab_cell_label[j] * torch.log(ab_cell_pred[j] + 1e-8) + 
                             (1. - ab_cell_label[j]) * torch.log((1. - ab_cell_pred[j]) + + 1e-8 )))
            
            cell_loss_ab.append(torch.where(torch.ne(ab_cell_label[j], -1.), bce_ab[j], torch.zeros(bce_ab[j].shape).to(self.device)))

        return self.abnormal*torch.stack(cell_loss_ab).mean() + torch.stack(cell_loss).mean(), cell_loss      


#         for j in range(batch_size) :

#             # bce for normal cell
#             bce.append(-(cell_label[j] * torch.log(cell_pred[j] + 1e-8) + (1. - cell_label[j]) * torch.log((1. - cell_pred[j]) + 1e-8)))
#             alpha = torch.where(torch.eq(cell_label[j], 1.), self.alpha, 1- self.alpha)
#             focal = torch.where(torch.eq(cell_label[j], 1.), 1.- cell_pred[j], cell_pred[j])
#             focal_weight = alpha * torch.pow(focal, self.gamma)
#             cell_loss_bce = focal_weight * bce[j]
#             cell_loss.append(torch.where(torch.ne(cell_label[j], -1.), cell_loss_bce, torch.zeros(cell_loss_bce.shape).to(self.device)))

#             # bce for abnormal cell
#             bce_ab.append( -(ab_cell_label[j] * torch.log(ab_cell_pred[j] + 1e-8) + 
#                              (1. - ab_cell_label[j]) * torch.log((1. - ab_cell_pred[j]) + + 1e-8 )))
#             alpha = torch.where(torch.eq(ab_cell_label[j], 1.), self.alpha, 1- self.alpha)
#             focal = torch.where(torch.eq(ab_cell_label[j], 1.), 1.- ab_cell_pred[j], ab_cell_pred[j])
#             focal_weight = alpha * torch.pow(focal, self.gamma)
#             cell_loss_bce_ab = focal_weight * bce_ab[j]
#             cell_loss_ab.append(cell_loss_bce_ab)       

#         return torch.stack(cell_loss_ab).mean() + torch.stack(cell_loss).mean()
        
