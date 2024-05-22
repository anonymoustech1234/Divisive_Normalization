"""
Return dataloaders

"""

import torch
from torch.utils.data import DataLoader, random_split

from torchvision import transforms, datasets

from typing import Tuple
import numpy
import random

def get_dataloaders(args_data: dict):

    # return train_loaders, val_loaders, test_loaders
    if args_data['dataset'] == 'imagenet':
        if args_data['gpu']:
            train_loader, val_loader, test_loader = get_dataloaders_gpu(args_data)
            return train_loader, val_loader, test_loader
        import modules.data.imagenet as dataset_helper
    elif args_data['dataset'] == 'cifar100':
        import modules.data.cifar100 as dataset_helper
    elif args_data['dataset'] == 'custom':
        import modules.data.custom as dataset_helper
    else:
        raise NotImplementedError(f"The {args_data['dataset']} dataset is not implemented")

    # create dataset
    train_set, val_set, test_set = dataset_helper.get_datasets(args_data)

    # create dataloaders
    train_loader = DataLoader(
        train_set, 
        args_data['bz'], 
        shuffle=True, 
        num_workers=args_data['num_workers'],
        worker_init_fn=seed_worker
    ) if train_set else None
    val_loader = DataLoader(
        val_set, 
        args_data['bz'], 
        shuffle=False, 
        num_workers=args_data['num_workers'],
        worker_init_fn=seed_worker
    ) if val_set else None
    test_loader = DataLoader(
        test_set, 
        args_data['bz'], 
        shuffle=False, 
        num_workers=args_data['num_workers'],
        worker_init_fn=seed_worker
    ) if test_set else None

    return train_loader, val_loader, test_loader

def get_dataloaders_gpu(args_data: dict):
    # specifically for gpu
    import modules.data.imagenet_gpu

    return modules.data.imagenet_gpu.get_dataloaders(args_data)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)