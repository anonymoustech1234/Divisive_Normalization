import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, args_model):
        super(VGG16, self).__init__()
        self.features = nn.Sequential()
        # Convolutional layers
        self.features.add_module('conv1_1', nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1))
        self.features.add_module('relu1_1', nn.ReLU(inplace=True))
        self.features.add_module('conv1_2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        self.features.add_module('relu1_2', nn.ReLU(inplace=True))
        self.features.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.features.add_module('conv2_1', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
        self.features.add_module('relu2_1', nn.ReLU(inplace=True))
        self.features.add_module('conv2_2', nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
        self.features.add_module('relu2_2', nn.ReLU(inplace=True))
        self.features.add_module('pool2', nn.MaxPool2d(kernel_size=2, stride=2))

        self.features.add_module('conv3_1', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
        self.features.add_module('relu3_1', nn.ReLU(inplace=True))
        self.features.add_module('conv3_2', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        self.features.add_module('relu3_2', nn.ReLU(inplace=True))
        self.features.add_module('conv3_3', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        self.features.add_module('relu3_3', nn.ReLU(inplace=True))
        self.features.add_module('pool3', nn.MaxPool2d(kernel_size=2, stride=2))

        self.features.add_module('conv4_1', nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1))
        self.features.add_module('relu4_1', nn.ReLU(inplace=True))
        self.features.add_module('conv4_2', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.features.add_module('relu4_2', nn.ReLU(inplace=True))
        self.features.add_module('conv4_3', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.features.add_module('relu4_3', nn.ReLU(inplace=True))
        self.features.add_module('pool4', nn.MaxPool2d(kernel_size=2, stride=2))

        self.features.add_module('conv5_1', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.features.add_module('relu5_1', nn.ReLU(inplace=True))
        self.features.add_module('conv5_2', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.features.add_module('relu5_2', nn.ReLU(inplace=True))
        self.features.add_module('conv5_3', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.features.add_module('relu5_3', nn.ReLU(inplace=True))
        self.features.add_module('pool5', nn.MaxPool2d(kernel_size=2, stride=2))

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1000)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_layers(self) -> dict:
        # TODO: Return important weight
        output_dict = {}

        output_dict['conv1'] = self.features.conv1_1

        return output_dict

