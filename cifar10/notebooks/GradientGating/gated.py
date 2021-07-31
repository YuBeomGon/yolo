import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np
        
# normal case, stoachastic gradient 
class Conv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
        # Save arguments to context to use on backward
        # WARNING : if stride, padding, dilation etc is array, this will not work properly!!!!
#         print('stride', stride)
        if weight.shape[2] == 1 :
            padding = 0
        elif weight.shape[2] == 5 :
            padding = 2
        elif weight.shape[2] == 7 :
            padding = 3
        confs = torch.from_numpy(np.array([stride, padding, dilation, groups]))
        out = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
#         print(out.shape)
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
#         grad_output = grad_output * 2*torch.sigmoid(out)
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
class Conv2dFunctionG(Conv2dFunction):
    
    @staticmethod
    def backward(ctx, grad_output):
        # Load saved tensors
        input, out, weight, bias, confs = ctx.saved_variables
        confs = confs.numpy()
        stride, padding, dilation, groups= confs[0], confs[1], confs[2], confs[3]

        # Calculate Gradient
        grad_input = grad_weight = grad_bias = None
        
#         gradient is gated according to the feature map of each layer
#         batch_size, channel, k_h, k_w = out.shape
#         denom = torch.norm(out.view(batch_size, -1), dim=1).unsqueeze(dim=1)/torch.sqrt(torch.tensor(k_h * k_w))
#         out = torch.abs((out.view(batch_size, -1) / denom).view(batch_size, channel, k_h, k_w))

#         grad_output = grad_output * 2 * torch.sigmoid(out/max_out)
#         grad_output1 = grad_output * torch.nn.ReLU6()(out)

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)
#             grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output1, stride, padding, dilation, groups)
#             grad_input = 2*torch.sigmoid(input)*torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)
            
        if ctx.needs_input_grad[1]:
            grad_output = grad_output * 2 * torch.sigmoid(out)
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)
                
        # WARNING : Bias maybe buggy, remove if it is buggy
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        # WARNING : Bias maybe buggy, remove if it is buggy
        if bias is not None:
            return grad_input, grad_weight, grad_bias, None, None, None, None
        else:
            return grad_input, grad_weight, None, None, None, None, None        

# Gated gradient version2(model 2, bn relu change)
class Conv2dFunctionG2(Conv2dFunction):
    
    @staticmethod
    def backward(ctx, grad_output):
        # Load saved tensors
        input, out, weight, bias, confs = ctx.saved_variables
        confs = confs.numpy()
        stride, padding, dilation, groups= confs[0], confs[1], confs[2], confs[3]

        # Calculate Gradient
        grad_input = grad_weight = grad_bias = None
        
#         gradient is gated according to the feature map of each layer
#         grad_output = grad_output * 2*torch.sigmoid(out)
        grad_output = grad_output * 2*(torch.sigmoid(torch.abs(out)))
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)
#             grad_input = 2*torch.sigmoid(input)*torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)
            
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
                

# https://github.com/VictorZuanazzi/DeepLearning201/blob/master/assignment_1/code/custom_batchnorm.py        
class CustomBatchNormManualFunction(torch.autograd.Function):
    """
    This torch.autograd.Function implements a functional custom version of the 
    batch norm operation for MLPs.
    Using torch.autograd.Function allows you to write a custom backward function.
    The function will be called from the nn.Module CustomBatchNormManualModule
    Inside forward the tensors are (automatically) not recorded for automatic 
    differentiation since the backward pass is done via the backward method.
    The forward pass is not called directly but via the apply() method. This 
    makes sure that the context objects  are dealt with correctly. 
    Example:
    my_bn_fct = CustomBatchNormManualFunction()
    normalized = fct.apply(input, gamma, beta, eps)
    """
    
    def __init__(self, insize,  momentum=0.9) :
        self.running_mean = torch.zeros(insize)
        self.running_var = torch.ones(insize)
        self.momentum = momentum
        self.device = torch.device('cuda')

    @staticmethod
    def forward(ctx, input, gamma, beta, eps=1e-5):
        """
        Compute the batch normalization

        Args:
          ctx: context object handling storing and retrival of tensors and constants 
              and specifying whether tensors need gradients in backward pass
          input: input tensor of shape (n_batch, n_neurons)
          gamma: variance scaling tensor, applied per neuron, shpae (n_neurons)
          beta: mean bias tensor, applied per neuron, shpae (n_neurons)
          eps: small float added to the variance for stability
        Returns:
          out: batch-normalized tensor
        TODO:
          Implement the forward pass of batch normalization
          Store constant non-tensor objects via ctx.constant=myconstant
          Store tensors which you need in the backward pass via 
              ctx.save_for_backward(tensor1, tensor2, ...)
          Intermediate results can be decided to be either recomputed in the 
              backward pass or to be stored for the backward pass. Do not store 
              tensors which are unnecessary for the backward pass to save memory!
          For the case that you make use of torch.var be aware that the flag 
              unbiased=False should be set.
        """
    
        nbatch, channel, height, width = input.shape
        #forward pass
#         mu = input.mean(dim=0)\
        mu = input.mean([0,2,3])

        #variance
#         var = input.var(dim=0, unbiased = False)
        var = input.var([0,2,3], unbiased = False)
    
#         running_mean_current = self.momentum * self.running_mean
#         running_mean_current = running_mean_current.to(self.device)
#         self.running_mean = running_mean_current + (1.0-self.momentum) * mean
        
#         running_var_current = self.momentum * self.running_var
#         running_var_current = running_var_current.to(self.device)
#         self.running_var = running_var_current + (1.0-self.momentum) * (input.shape[0]/(input.shape[0]-1)*var)

        #normalization
        center_input = input - mu.view([1, channel, 1,1]).expand_as(input)
        denominator = var.view([1, channel, 1,1]).expand_as(input) + eps
        denominator = denominator.sqrt()

        in_hat = center_input/denominator

        #scale and shift
        out = (gamma * in_hat + beta).view(nbatch,channel,height,width)

        #store constants
        ctx.save_for_backward(gamma, denominator, in_hat)
        ctx.epsilon = eps

        return out


    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute backward pass of the batch normalization.

        Args:
          ctx: context object handling storing and retrival of tensors and 
              constants and specifying whether tensors need gradients in backward pass
        Returns:
          out: tuple containing gradients for all input arguments

        TODO:
          Retrieve saved tensors and constants via ctx.saved_tensors and ctx.constant
          Compute gradients for inputs where ctx.needs_input_grad[idx] is True. 
          Set gradients for other inputs to None. This should be decided dynamically.
        """

        #get dimensions
        batch_size, n_channel, height, width  = grad_output.shape

        #get useful parameters stored in the forward pass
        #gamma, mu, center_input, var, denominator, in_hat = ctx.saved_tensors
        gamma, denominator, in_hat = ctx.saved_tensors

        #to avoid unnecessary matrix inversions 
        den_inv = 1/denominator

        #gradient of the input
        if ctx.needs_input_grad[0]:
            grad_output = grad_output.view(batch_size, height, width, n_channel)

            grad_in_hat = grad_output * gamma

            term_1 = batch_size * grad_in_hat
            term_2 = torch.sum(grad_in_hat, dim=0)
            term_3 = in_hat * torch.sum(grad_in_hat * in_hat, dim=0)

            grad_input = ((1/batch_size) * den_inv * (term_1 - term_2 - term_3)).view(batch_size, n_channel, height, width)

        else:
            grad_input = None


        #gradient of gamma
        if ctx.needs_input_grad[1]:
            grad_gamma = torch.sum(torch.mul(grad_output, in_hat), dim=0)
        else:
            grad_gamma = None

        #gradient of beta
        if ctx.needs_input_grad[2]:
            grad_beta = grad_output.sum(dim=0)
        else:
            grad_beta = None


        # return gradients of the three tensor inputs and None for the constant eps
        return grad_input, grad_gamma, grad_beta, None        
    
# normal model
class Net(nn.Module):
    def __init__(self, conv_list, conv):
        super(Net, self).__init__()
        self.conv1 = conv.apply
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = conv.apply
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = conv.apply
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = conv.apply
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = conv.apply
        self.avgpool = torch.nn.AvgPool2d((2,2) ,stride=(2,2))
        self.maxpool = torch.nn.MaxPool2d((2,2), stride=(2,2))
        self.linear = torch.nn.Linear(128, 10)
        self.act = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.device = torch.device("cuda")
        self.dtype = torch.float
        self.c1 = None
        self.c2 = None
        self.c3 = None
        self.c4 = None
        self.c5 = None
        
        # weights of batch norm for custom backward, normal case 
        self.conw1, self.conw2, self.conw3, self.conw4, self.conw5 = conv_list 

    def forward(self, x):

        self.c1 = self.conv1(x, self.conw1)
        x = self.bn1(self.act(self.c1))
        x = self.maxpool(x)
        self.c2 = self.conv2(x, self.conw2)
        x = self.bn2(self.act(self.c2))
        x = self.maxpool(x)
        self.c3 = self.conv3(x, self.conw3)
        x = self.bn3(self.act(self.c3))
        x = self.maxpool(x)
        self.c4 = self.conv4(x, self.conw4)
        x = self.bn4(self.act(self.c4))
        
        x = self.maxpool(x)
        self.c5 = self.conv5(x, self.conw5)
        x = self.avgpool(self.c5)
        x = torch.squeeze(x)
#         x = self.linear(x)
        x = torch.nn.Softmax(dim=1)(x)
#         x = torch.sigmoid(x)
        
        return x
    
    def conv_return (self) :
        return self.c1, self.c2, self.c3, self.c4, self.c5

# gated model    
class Net1(Net):

    def forward(self, x):
        self.c1 = self.conv1(x, self.conw1)
        x = self.act(self.bn1(self.c1))
        x = self.maxpool(x)
        self.c2 = self.conv2(x, self.conw2)
        x = self.act(self.bn2(self.c2))
        x = self.maxpool(x)
        self.c3 = self.conv3(x, self.conw3)
        x = self.act(self.bn3(self.c3))
        x = self.maxpool(x)
        self.c4 = self.conv4(x, self.conw4)
        x = self.act(self.bn4(self.c4))
        
        x = self.maxpool(x)
        self.c5 = self.conv5(x, self.conw5)
        x = self.avgpool(self.c5)
        x = torch.squeeze(x)
#         x = self.linear(x)
        x = torch.nn.Softmax(dim=1)(x)
#         x = torch.sigmoid(x)
        
        return x    
    
class NetS(Net):

    def forward(self, x):
        x = self.tanh(self.conv1(x, self.conw1)/5)
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.tanh(self.conv2(x, self.conw2)/5)
        x = self.bn2(x)
        x = self.maxpool(x)
        x = self.tanh(self.conv3(x, self.conw3)/5)
        x = self.bn3(x)
        x = self.maxpool(x)
        x = self.tanh(self.conv4(x, self.conw4)/5)
        x = self.bn4(x)
        
        x = self.maxpool(x)
        x = self.conv5(x, self.conw5)
        x = self.avgpool(x)
        x = torch.squeeze(x)
#         x = self.linear(x)
        x = torch.nn.Softmax(dim=1)(x)
#         x = torch.sigmoid(x)
        
        return x      
    

#     adam model
class NetA(nn.Module):
    def __init__(self):
        super(NetA, self).__init__()
        self.conv1 = nn.Conv2d(3,8,kernel_size=(5,5),padding=2,bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8,32,kernel_size=(3,3),padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,128,kernel_size=(3,3),padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,128,kernel_size=(3,3),padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128,10,kernel_size=(1,1),padding=0,bias=False)
        self.avgpool = torch.nn.AvgPool2d((2,2) ,stride=(2,2))
        self.maxpool = torch.nn.MaxPool2d((2,2), stride=(2,2))
#         self.linear = torch.nn.Linear(128, 10)
        self.act = torch.nn.ReLU()
        

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.act(x)
        x = self.maxpool(x)
        x = self.bn2(self.conv2(x))
        x = self.act(x)
        x = self.maxpool(x)
        x = self.bn3(self.conv3(x))
        x = self.act(x)
        x = self.maxpool(x)
        x = self.bn4(self.conv4(x))
        x = self.act(x)
        x = self.maxpool(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.squeeze(x)
#         x = self.linear(x)
        x = torch.nn.Softmax(dim=1)(x)
#         x = torch.sigmoid(x)
        
        return x            
    
