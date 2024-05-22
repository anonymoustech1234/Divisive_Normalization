import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual_block(nn.Module):
    # also named bottleneck block
    expansion = 4 # expansion ratio in the bottleneck architecture

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Residual_block, self).__init__()
        
        self.features = nn.Sequential()
        self.features.add_module('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        self.features.add_module('relu1', nn.ReLU(inplace=True))
        self.features.add_module('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        self.features.add_module('relu2', nn.ReLU(inplace=True))
        self.features.add_module('conv3', nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False))

        self.downsample = downsample

        self.activation = nn.Sequential()
        self.activation.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        identity = x

        out = self.features(x)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out

class Resnet50(nn.Module):
    def __init__(self, args_model):
        self.layers = [3, 4, 6, 3] # nums of blocks per layer for resnet50

        super(Resnet50, self).__init__()
        
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
        self.fc = nn.Linear(512 * Residual_block.expansion, args_model['num_classes'])

    def _make_layer(self, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * Residual_block.expansion:
            downsample = nn.Sequential()
            downsample.add_module('conv1', nn.Conv2d(
                self.in_channels, out_channels * Residual_block.expansion, kernel_size=1, stride=stride, bias=False
            ))

        layers = []
        layers.append(Residual_block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * Residual_block.expansion
        for _ in range(1, num_blocks):
            layers.append(Residual_block(self.in_channels, out_channels))

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