import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms, datasets

from typing import Tuple

def get_datasets(args_data):
    # return train / val / test datasets of cifar100 with pytorch io

    path = args_data['path']

    train_transform, val_transform, test_transform = get_transforms(args_data)

    train_set = datasets.CIFAR100(root=path, train=True, transform=train_transform, download=True)
    test_set = datasets.CIFAR100(root=path, train=False, transform=test_transform, download=True)

    train_subset, val_subset = train_val_split(train_set)
    val_subset.transform = val_transform

    return train_subset, val_subset, test_set

def train_val_split(train_set: Dataset) -> Tuple[Dataset, Dataset]:
    train_size = int(len(train_set) * 0.8)
    val_size = len(train_set) - train_size
    train_subset, val_subset = random_split(train_set, [train_size, val_size])
    # generator=torch.Generator().manual_seed(42)
    return train_subset, val_subset

def get_transforms(args_data):
    # return train / val / test transform for cifar100
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.5071, 0.4867, 0.4408],
            std = [0.2675, 0.2565, 0.2761]
        )
    ])

    return transform, transform, transform