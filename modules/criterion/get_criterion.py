from torch.nn import CrossEntropyLoss

def get_criterion(args_training):

    criterion = None

    if args_training['criterion'] == 'cross_entropy':
        criterion = CrossEntropyLoss()
    else:
        raise NotImplementedError(f"Criterion {args_training['criterion']} not implemented.")
    
    return criterion