from torch.optim import Adam, SGD

def get_optimizer(args_optimizer, model):

    optimizer = None
    if args_optimizer['name'] == "sgd":
        optimizer = SGD(
            filter(lambda p: p.requires_grad, model.parameters()), # is this universal？
            lr = args_optimizer['lr'],
            weight_decay = args_optimizer['weight_decay'],
            momentum = args_optimizer['momentum'],
            dampening = args_optimizer['dampening'],
            nesterov = args_optimizer['nesterov']
        )
    elif args_optimizer['name'] == 'adam':
        optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr = args_optimizer['lr'],
            weight_decay = args_optimizer['weight_decay'],
            betas = (args_optimizer['beta1'], args_optimizer['beta2']),
            eps = args_optimizer['eps'],
            amsgrad = args_optimizer['amsgrad']
        )
    else:
        raise NotImplementedError(f"Optimizer {args_optimizer['name']} not implemented")
    
    return optimizer