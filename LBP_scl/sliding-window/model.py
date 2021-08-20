"""
Implementation of YOLOv3 architecture
"""

import torch
import torch.nn as nn
from resnet import resnet12

class mlp_3layer(nn.Module):
    def __init__(self, in_channels = 1024, out_class=2):
        super().__init__() 
        self.mid_channels = int(in_channels/4)
        self.last_channels = int(self.mid_channels/4)
        self.fc1 = nn.Linear(in_channels, self.mid_channels, bias=True) 
        self.bn1 = nn.BatchNorm1d(self.mid_channels)
        self.fc2 = nn.Linear(self.mid_channels, self.last_channels, bias=True)
        self.bn2 = nn.BatchNorm1d(self.last_channels)
        self.fc3 = nn.Linear(self.last_channels, out_class, bias=True)
        self.bn3 = nn.BatchNorm1d(out_class)
        self.act = nn.LeakyReLU()     

    def forward(self, x, indices=None):
#         print(x.shape)
        batch_size, channels, _, _ = x.shape
        x = torch.flatten(x, start_dim=2, end_dim=3)
        x = x.permute(0,2,1)
#         x = x.view(batch_size, -1, channels)
        x = self.fc1(x)
        x = x.permute(0,2,1)
        x = self.bn1(x)
        x = x.permute(0,2,1)
        x = self.act(x)
#         x = self.act(self.bn1(self.fc1(x)))
#         print(x.shape)
        x = self.fc2(x)
        x = x.permute(0,2,1)
        x = self.bn2(x)
        x = x.permute(0,2,1)
        x = self.act(x)
        
        x = self.fc3(x)
#         x = x.permute(0,2,1)
#         x = self.bn3(x)
#         x = x.permute(0,2,1)        
        x = torch.sigmoid(x)
      
        return x

class LBPModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, kernel_size=64, stride=32, isCNN=False):
        super().__init__()
        self.num_classes = num_classes # normal, abnormal
        self.in_channels = in_channels
        self.out_channels = 1024
        self.feature_extractor = nn.Conv2d(in_channels, out_channels=self.out_channels, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.act = nn.LeakyReLU()
        self.mlp = mlp_3layer(in_channels=self.out_channels, out_class=num_classes)
        self.cnn = cnn_layer(in_channels=self.out_channels, out_class=num_classes)
        self.isCNN = isCNN
        
    def forward(self, x):    
        x = self.act(self.bn(self.feature_extractor(x)))
        print('feature', x.shape)
        if self.isCNN :
            x = self.cnn(x)
        else :
            x = self.mlp(x)
        
        return x
    
class CNNModel(nn.Module) :
    def __init__(self, kernel_size=64, stride=32) :
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.unfold = nn.Unfold(kernel_size=(kernel_size, kernel_size), stride=stride)
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(32)        
        self.act = torch.nn.ReLU()
        self.conv1x1 = nn.Conv2d(32, 2, kernel_size=1, bias=True)
        self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.pool2 = nn.MaxPool2d((4, 4), stride=(4, 4))
        
#     print(patch.view(3,64,64).permute(1,2,0)[1])
    def forward (self, x, indices=None) :
        batch_size, _, _, _ = x.shape 
#         print(x.shape)
        x = self.unfold(x)
        x = x.view(batch_size, 3, self.kernel_size, self.kernel_size, -1).permute(0,4,2,3,1)
        split_x = torch.split(x, 1, dim=0)
        batch_x = []
        
        for x in split_x :
            x = self.conv1(x.squeeze().permute(0,3,1,2))
            x = self.bn1(self.act(x))
            x = self.pool1(x)
            
            x = self.conv2(x)
            x = self.bn2(self.act(x))
            x = self.pool1(x)
            
            x = self.conv3(x)
            x = self.bn3(self.act(x))
            x = self.pool1(x)  
            
            x = self.conv4(x)
            x = self.bn4(self.act(x))
            x = self.pool1(x)  
            
            x = self.conv5(x)
            x = self.bn5(self.act(x))
            x = self.pool2(x)   
            
            x = self.conv1x1(x).squeeze().unsqueeze(dim=0)
            x = torch.sigmoid(x)
#             print(x.shape)
            batch_x.append(x)
        
#         x = self.conv1(x.view(-1,self.kernel_size,self.kernel_size,3))
#         print(x.shape)
        
        return torch.cat(batch_x)

# class CNNBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.leaky = nn.LeakyReLU(0.1)
#         self.use_bn_act = bn_act

#     def forward(self, x):
#         if self.use_bn_act:
#             return self.leaky(self.bn(self.conv(x)))
#         else:
#             return self.conv(x)


# class ResidualBlock(nn.Module):
#     def __init__(self, channels, use_residual=True, num_repeats=1):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         for repeat in range(num_repeats):
#             self.layers += [
#                 nn.Sequential(
#                     CNNBlock(channels, channels // 2, kernel_size=1),
#                     CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
#                 )
#             ]

#         self.use_residual = use_residual
#         self.num_repeats = num_repeats

#     def forward(self, x):
#         for layer in self.layers:
#             if self.use_residual:
#                 x = x + layer(x)
#             else:
#                 x = layer(x)

#         return x


# class ScalePrediction(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super().__init__()
#         self.pred = nn.Sequential(
#             CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
#             CNNBlock(
#                 2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
#             ),
#         )
#         self.num_classes = num_classes

#     def forward(self, x):
#         return (
#             self.pred(x)
#             .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
#             .permute(0, 1, 3, 4, 2)
#         )


# class LBPModel(nn.Module):
#     def __init__(self, in_channels=3, num_classes=80):
#         super().__init__()
#         self.num_classes = num_classes
#         self.in_channels = in_channels
#         self.layers = self._create_conv_layers()

#     def forward(self, x):
#         outputs = []  # for each scale
#         route_connections = []
#         for layer in self.layers:
#             if isinstance(layer, ScalePrediction):
#                 outputs.append(layer(x))
#                 continue

#             x = layer(x)

#             if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
#                 route_connections.append(x)

#             elif isinstance(layer, nn.Upsample):
#                 x = torch.cat([x, route_connections[-1]], dim=1)
#                 route_connections.pop()

#         return outputs

#     def _create_conv_layers(self):
#         layers = nn.ModuleList()
#         in_channels = self.in_channels

#         for module in config:
#             if isinstance(module, tuple):
#                 out_channels, kernel_size, stride = module
#                 layers.append(
#                     CNNBlock(
#                         in_channels,
#                         out_channels,
#                         kernel_size=kernel_size,
#                         stride=stride,
#                         padding=1 if kernel_size == 3 else 0,
#                     )
#                 )
#                 in_channels = out_channels

#             elif isinstance(module, list):
#                 num_repeats = module[1]
#                 layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

#             elif isinstance(module, str):
#                 if module == "S":
#                     layers += [
#                         ResidualBlock(in_channels, use_residual=False, num_repeats=1),
#                         CNNBlock(in_channels, in_channels // 2, kernel_size=1),
#                         ScalePrediction(in_channels // 2, num_classes=self.num_classes),
#                     ]
#                     in_channels = in_channels // 2

#                 elif module == "U":
#                     layers.append(nn.Upsample(scale_factor=2),)
#                     in_channels = in_channels * 3

#         return layers


if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")