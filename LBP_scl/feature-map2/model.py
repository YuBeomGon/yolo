import torch
import torch.nn as nn

class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=1, feature_size=256):
        super(RegressionModel, self).__init__()
        
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.LeakyReLU(0.1)

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.LeakyReLU(0.1)

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=1, padding=0)
        self.act4 = nn.LeakyReLU(0.1)

#         center point x, y
        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=1, padding=0)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)
        
#         x, y, width, height
        return out


class CellClassification(nn.Module):
    def __init__(self, num_features_in, num_anchors=1, num_classes=1, feature_size=256):
        super(CellClassification, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.LeakyReLU(0.1)

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.LeakyReLU(0.1)

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=1, padding=0)
        self.act4 = nn.LeakyReLU(0.1)

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=1, padding=0)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        return out


class ABCellClassification(nn.Module):
    def __init__(self, num_features_in, num_anchors=1, num_classes=1, feature_size=256):
        super(ABCellClassification, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.LeakyReLU(0.1)

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.LeakyReLU(0.1)

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=1, padding=0)
        self.act4 = nn.LeakyReLU(0.1)

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=1, padding=0)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        return out
    
class MultiClassification(nn.Module):
    def __init__(self, num_features_in, num_anchors=1, num_classes=1, feature_size=256):
        super(MultiClassification, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.LeakyReLU(0.1)

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.LeakyReLU(0.1)

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=1, padding=0)
        self.act4 = nn.LeakyReLU(0.1)

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=1, padding=0)
        self.output_act = nn.Sigmoid()
#         self.output_act = nn.Softmax()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        return out    

""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""

# tuple CNNblock
# list Residual block
# S scale prediction
# U upsample
config = [
    (16, 5, 1),
    (32, 3, 2),
    ["B", 1],
    (64, 3, 2),
    ["B", 2],
    (128, 3, 2),
    ["B", 2],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 2],    
#     (1024, 3, 2),
#     ["B", 4],  # To this point is Darknet-53
]
    
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x    
    
class Darknet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()
        self.CellClassification = CellClassification(256, num_classes=1)
        self.ABCellClassification = ABCellClassification(256, num_classes=1)
        self.MultiClassification = MultiClassification(256, num_classes=1)
        self.RegressionModel = RegressionModel(256)
    
    def forward(self, x, indices=None):
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            x = layer(x)
        
        x1 = self.CellClassification(x)
        x2 = self.ABCellClassification(x)
        x3 = self.MultiClassification(x)
        x4 = self.RegressionModel(x)

        return x1, x2, x3, x4

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

        return layers    
    
    
