"""
Functions to create datasets in a specific folder.

1. Create dataset object
2. Create transforms
"""

from torchvision import transforms, datasets
import os


def get_datasets(args_data: dict):
    # return the custom dataset

    # check if the folder exists
    if not os.path.exists(args_data['path']):
        raise FileNotFoundError(f"Custom dataset {args_data['dataset']} does not exist")
    
    transform = get_transforms(args_data)
    dataset = datasets.ImageFolder(root=args_data['path'], transform=transform)

    return None, None, dataset # Two None here stands for placeholder


def get_transforms(args_data: dict):
    # return the transform
    # TODO: consider do we need the expand for more datasets?

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])

    return transform