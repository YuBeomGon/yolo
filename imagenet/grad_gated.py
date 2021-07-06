import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np

# Gated gradient
class FeatureExtractor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=2, padding=3, dilation=1, groups=1):
        # Save arguments to context to use on backward
        # WARNING : if stride, padding, dilation etc is array, this will not work properly!!!!
        confs = torch.from_numpy(np.array([stride, padding, dilation, groups]))
        out = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        ctx.save_for_backward(input, out, weight, bias, confs)

        # Compute Convolution
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # Load saved tensors
        input, out, weight, bias, confs = ctx.saved_variables
        confs = confs.numpy()
        stride, padding, dilation, groups= confs[0], confs[1], confs[2], confs[3]

        # Calculate Gradient
        grad_input = grad_weight = grad_bias = None
        
#         gradient is gated according to the feature map of each layer
        grad_output = grad_output * 2*torch.sigmoid(out)
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)
            
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)
                
        # WARNING : Bias maybe buggy, remove if it is buggy
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)


        # WARNING : Bias maybe buggy, remove if it is buggy
        if bias is not None:
            return grad_input, grad_weight, grad_bias, None, None, None, None
        else:
            return grad_input, grad_weight, None, None, None, None, None
        
# Gated gradient
class FeatureExtractor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=2, padding=3, dilation=1, groups=1):
        # Save arguments to context to use on backward
        # WARNING : if stride, padding, dilation etc is array, this will not work properly!!!!
        confs = torch.from_numpy(np.array([stride, padding, dilation, groups]))
        out = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        ctx.save_for_backward(input, out, weight, bias, confs)

        # Compute Convolution
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # Load saved tensors
        input, out, weight, bias, confs = ctx.saved_variables
        confs = confs.numpy()
        stride, padding, dilation, groups= confs[0], confs[1], confs[2], confs[3]

        # Calculate Gradient
        grad_input = grad_weight = grad_bias = None
        
#         gradient is gated according to the feature map of each layer
        grad_output = grad_output * 2*torch.sigmoid(out)
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)
            
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)
                
        # WARNING : Bias maybe buggy, remove if it is buggy
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)


        # WARNING : Bias maybe buggy, remove if it is buggy
        if bias is not None:
            return grad_input, grad_weight, grad_bias, None, None, None, None
        else:
            return grad_input, grad_weight, None, None, None, None, None        
        
# from torch.autograd import gradcheck
# conv = Conv2dFunction.apply
# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
# input = (torch.randn(20,20,dtype=torch.double,requires_grad=True), torch.randn(30,20,dtype=torch.double,requires_grad=True))
# test = gradcheck(linear, input, eps=1e-6, atol=1e-4)
# print(test)        

# conw1 = torch.randn(8,3,5,5, device=device, dtype=dtype, requires_grad=True)
# conw2 = torch.randn(32,8,3,3, device=device, dtype=dtype, requires_grad=True)
# conw3 = torch.randn(128,32,3,3, device=device, dtype=dtype, requires_grad=True)
# conw4 = torch.randn(128,128,3,3, device=device, dtype=dtype, requires_grad=True)
# conw5 = torch.randn(10,128,1,1, device=device, dtype=dtype, requires_grad=True)

# # weight for normal
# conw1 = torch.nn.init.xavier_uniform_(conw1, gain=1.0)
# conw2 = torch.nn.init.xavier_uniform_(conw2, gain=1.0)
# conw3 = torch.nn.init.xavier_uniform_(conw3, gain=1.0)
# conw4 = torch.nn.init.xavier_uniform_(conw4, gain=1.0)
# conw5 = torch.nn.init.xavier_uniform_(conw5, gain=1.0)

# gated model    
class NetG(nn.Module):
    def __init__(self):
        super(NetG, self).__init__()
        self.conv1 = Conv2dFunctionG.apply
        self.conv2 = Conv2dFunctionG.apply
        self.conv3 = Conv2dFunctionG.apply
        self.conv4 = Conv2dFunctionG.apply
        self.conv5 = Conv2dFunctionG.apply
        self.avgpool = torch.nn.AvgPool2d((2,2) ,stride=(2,2))
        self.maxpool = torch.nn.MaxPool2d((2,2), stride=(2,2))
        self.linear = torch.nn.Linear(128, 10)
        self.act = torch.nn.ReLU()

    def forward(self, x, w1, w2, w3, w4, w5):
        x = self.act(self.conv1(x, w1))
        x = torch.nn.BatchNorm2d(8).to(device)(x)
        x = self.maxpool(x)
        x = self.act(self.conv2(x, w2))
        x = torch.nn.BatchNorm2d(32).to(device)(x)
        x = self.maxpool(x)
        x = self.act(self.conv3(x, w3))
        x = torch.nn.BatchNorm2d(128).to(device)(x)
        x = self.maxpool(x)
        x = self.act(self.conv4(x, w4))
        x = torch.nn.BatchNorm2d(128).to(device)(x)
        x = self.maxpool(x)
        x = self.conv5(x, w5)
        x = self.avgpool(x)
        x = torch.squeeze(x)
#         x = self.linear(x)
        x = torch.nn.Softmax(dim=1)(x)
#         x = torch.sigmoid(x)
        
        return x   

def variable (num, device='cuda', dtype=torch.float) :
    device=torch.device(device)
    out_channel = num[0]
    in_channel = num[1]
    kernel_height = num[2]
    kernel_width = num[3]
    weight = torch.randn(out_channel,in_channel,kernel_height,kernel_width,
            device=device, dtype=dtype, requires_grad=True )
    return torch.nn.init.xavier_uniform_(weight, gain=1.0)
 