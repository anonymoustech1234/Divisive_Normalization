import torch
import torch.nn as nn

class Alexnet(nn.Module):
    def __init__(self, args_model=None) -> None:
        super().__init__()

        dataset = args_model['for_dataset']

        self.features = nn.Sequential()
        if dataset == 'cifar100':
            self.features.add_module('conv1', nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1))
        else:
            self.features.add_module('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2))
        self.features.add_module('relu1', nn.ReLU(inplace=True))
        self.features.add_module('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2))
        if dataset == 'cifar100':
            self.features.add_module('conv2', nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1))
        else:
            self.features.add_module('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2))
        self.features.add_module('relu2', nn.ReLU(inplace=True))
        self.features.add_module('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2))
        self.features.add_module('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1))
        self.features.add_module('relu3', nn.ReLU(inplace=True))
        self.features.add_module('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1))
        self.features.add_module('relu4', nn.ReLU(inplace=True))
        self.features.add_module('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.features.add_module('maxpool3', nn.MaxPool2d(kernel_size=3, stride=2))

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, args_model['num_classes']),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_layers(self) -> dict:
        # Return important weight

        output_dict = {}

        output_dict['conv1'] = self.features.conv1
        output_dict['conv2'] = self.features.conv2
        output_dict['conv3'] = self.features.conv3
        output_dict['conv4'] = self.features.conv4
        output_dict['conv5'] = self.features.conv5

        return output_dict
    