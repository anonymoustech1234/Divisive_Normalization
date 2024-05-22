"""
This file includes functions about changing alexnet.
TODO: More init methods.
"""

import torch
import torch.nn as nn
from structures.get_divnorm import get_divnorm
from torch.distributions.multivariate_normal import MultivariateNormal

import math

def delete_relu(model: nn.Module, args_model: dict) -> nn.Module:
    # delete relu layers

    new_features = nn.Sequential()
    for name, module in model.features.named_children():
        if isinstance(module, nn.ReLU):
            continue
        else:
            new_features.add_module(name, module)
    model.features = new_features
    # consider the old features, will garbage collecter delete it?

    return model

def add_relu(model: nn.Module, args_model: dict) -> nn.Module:
    # add relu layers right after conv2d

    new_features = nn.Sequential()
    for name, module in model.features.named_children():
        new_features.add_module(name, module)
        if isinstance(module, nn.Conv2d):
            # insert a batchnorm2d after every conv2d
            relu_name = name.replace('conv', 'relu')
            new_features.add_module(relu_name, nn.ReLU(inplace=True))
    model.features = new_features
    # consider the old features, will garbage collecter delete it?
    
    return model

def add_batchnorm(model: nn.Module, args_model: dict) -> nn.Module:
    # insert batchnorm layers right after conv2d
    
    new_features = nn.Sequential()
    for name, module in model.features.named_children():
        new_features.add_module(name, module)
        if isinstance(module, nn.Conv2d):
            # insert a batchnorm2d after every conv2d
            bn_name = name.replace('conv', 'bn')
            new_features.add_module(bn_name, nn.BatchNorm2d(module.out_channels))
    model.features = new_features
    # consider the old features, will garbage collecter delete it?
    
    return model


def add_divnorm(model: nn.Module, args_model: dict) -> nn.Module:
    # insert divnorm layers right after conv2d
    
    new_features = nn.Sequential()
    for name, module in model.features.named_children():
        new_features.add_module(name, module)
        if isinstance(module, nn.Conv2d):
            # insert a divnorm2d after every conv2d
            dn_name = name.replace('conv', 'divnorm')
            dn = get_divnorm(
                module.out_channels, 
                args_model['divnorm_specs']
            )
            new_features.add_module(dn_name, dn)
    model.features = new_features
    # consider the old features, will garbage collecter delete it?
    
    return model

def init_weights(model: nn.Module, args_model: dict) -> nn.Module:
    # init weights (of convolutional layers)
    
    # in alexnet, conv2d was stored within a nn.sequential() object
    if args_model['weight_init'] == 'default':
        print('Init weight: default')

    elif args_model['weight_init'] == 'kaiming':
        print('Init weight: kaiming')
        nonlinearity = args_model['nonlinearity']
        for _, module in model.features.named_children():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    a = 0,
                    mode = 'fan_in',
                    nonlinearity = nonlinearity
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    elif args_model['weight_init'] == 'normal':
        print("Init weight: normal")
        for _, module in model.features.named_children():
            if isinstance(module, nn.Conv2d):
                if args_model['std'] == 'input_size':
                    std = 1 / ((module.weight.shape[1] * module.weight.shape[2] * module.weight.shape[3]) ** 0.5)
                else:
                    std = args_model['std']
                nn.init.normal_(
                    module.weight,
                    mean = 0.0,
                    std = std
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    elif args_model['weight_init'] == "multivariate_gaussian":
        p = float(args_model['divnorm_specs']['p'])
        print(f"Init weight: multivariate_gaussian with p = {p}")
        for _, module in model.features.named_children():
            if isinstance(module, nn.Conv2d):
                out_channels, in_channels, kernel_size, kernel_size = module.weight.data.shape
                filter_weights = None
                pool_size = 2**int(p * math.log2(out_channels))
                for i in range(out_channels // pool_size):

                    means = torch.normal(0, args_model['mean_std'], size=(in_channels * kernel_size * kernel_size,))

                    stds = torch.mul(torch.eye((in_channels * kernel_size * kernel_size)), args_model['std'])

                    samples = MultivariateNormal(means, stds).sample((pool_size,))
                    pool_weights = samples[:, :].reshape((-1, in_channels, kernel_size, kernel_size))
                    pool_weights.reqiures_grad = True
                    if filter_weights is None:
                        filter_weights = pool_weights
                    else:
                        filter_weights = torch.cat((filter_weights, pool_weights), dim=0)
                with torch.no_grad():
                    module.weight = nn.Parameter(filter_weights)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    elif args_model['weight_init'] == "kaiming_gaussian":
        p = float(args_model['divnorm_specs']['p'])
        print(f"Init weight: kaiming_gaussian with p = {p} and std ratio of {args_model['std_ratio']}")
        for _, module in model.features.named_children():
            if isinstance(module, nn.Conv2d):
                out_channels, in_channels, kernel_size, kernel_size = module.weight.data.shape
                filter_weights = None
                pool_size = 2**int(p * math.log2(out_channels))

                #get means:
                kaiming_samples = nn.init.kaiming_normal_(
                        module.weight,
                        a = 0,
                        mode = 'fan_in',
                        nonlinearity = 'relu'
                    )
                print(kaiming_samples.shape)
                for i in range(out_channels // pool_size):

                    means = (kaiming_samples[i]).flatten()
                    std_value = torch.std(kaiming_samples[i])*args_model['std_ratio']
                    stds = torch.mul(torch.eye((in_channels * kernel_size * kernel_size)), std_value)

                    samples = MultivariateNormal(means, stds).sample((pool_size,))
                    pool_weights = samples[:, :].reshape((-1, in_channels, kernel_size, kernel_size))
                    pool_weights.reqiures_grad = True
                    if filter_weights is None:
                        filter_weights = pool_weights
                    else:
                        filter_weights = torch.cat((filter_weights, pool_weights), dim=0)
                with torch.no_grad():
                    module.weight = nn.Parameter(filter_weights)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    else:
        raise NotImplementedError(f"init method {args_model['weight_init']} not implemented")
    
    return model
