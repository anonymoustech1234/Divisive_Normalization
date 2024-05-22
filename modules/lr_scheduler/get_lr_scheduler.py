import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

def get_lr_scheduler(args_lr_scheduler, optimizer):
    # THIS FUNCTION INPUT MIGHT NEED CONSIDERATION
    # How to input epochs? 
    
    lr_scheduler = None
    if args_lr_scheduler['name'] == 'cosinelr':
        lr_scheduler = CosineAnnealingLR(
            optimizer = optimizer, 
            T_max = args_lr_scheduler['t_max']
        )
    elif args_lr_scheduler['name'] == 'steplr':
        lr_scheduler = StepLR(
            optimizer = optimizer, 
            step_size = args_lr_scheduler['step_size'],
            gamma = args_lr_scheduler['gamma']
        )
    elif args_lr_scheduler['name'] == 'plateau':
        lr_scheduler = ReduceLROnPlateau(
            optimizer = optimizer,
            patience = args_lr_scheduler['patience'],
            factor = args_lr_scheduler['factor']
        )
    else:
        raise NotImplementedError(f"lr_scheduler {args_lr_scheduler['name']} not implemented")

    return lr_scheduler
