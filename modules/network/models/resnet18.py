import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Basic_block(nn.Module):
    # also named bottleneck block
    expansion = 1 # expansion ratio in the bottleneck architecture

    def __init__(self, in_channels, out_channels, stride=1):
        super(Basic_block, self).__init__()
        
        self.features = nn.Sequential()
        self.features.add_module('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        self.features.add_module('relu1', nn.ReLU(inplace=True))
        self.features.add_module('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut.add_module('conv1', nn.Conv2d(in_channels, out_channels * Basic_block.expansion, kernel_size=1, stride=stride, bias=False))

        self.activation = nn.Sequential()
        self.activation.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        identity = x

        out = self.features(x)
        identity = self.shortcut(x)

        out += identity
        out = self.activation(out)

        return out

class Resnet18(nn.Module):
    def __init__(self, args_model):
        self.layers = [2, 2, 2, 2] # nums of blocks per layer for resnet50

        super(Resnet18, self).__init__()
        
        self.in_channels = 64
        self.head = nn.Sequential()
        self.head.add_module('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False))
        self.head.add_module('relu1', nn.ReLU(inplace=True))
        self.head.add_module('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(out_channels=64, num_blocks=self.layers[0], stride=1)
        self.layer2 = self._make_layer(out_channels=128, num_blocks=self.layers[1], stride=2)
        self.layer3 = self._make_layer(out_channels=256, num_blocks=self.layers[2], stride=2)
        self.layer4 = self._make_layer(out_channels=512, num_blocks=self.layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Basic_block.expansion, args_model['num_classes'])

    def _make_layer(self, out_channels, num_blocks, stride):

        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(Basic_block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * Basic_block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.head(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    def get_layers(self) -> dict:
        # TODO: Return important weight
        output_dict = {}

        output_dict['conv1'] = self.head.conv1

        return output_dict