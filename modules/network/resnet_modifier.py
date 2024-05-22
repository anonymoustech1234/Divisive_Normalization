import torch
import torch.nn as nn
from structures.get_divnorm import get_divnorm
from modules.network.models.resnet50 import Resnet50
from modules.network.models.resnet18 import Resnet18
from torch.distributions.multivariate_normal import MultivariateNormal
import math

def delete_relu(model: nn.Module, args_model: dict):

    def delete_relu_in_sequential(sequential, args_model: dict):
        new_sequential = nn.Sequential()
        for name, module in sequential.named_children():
            if isinstance(module, nn.ReLU):
                continue
            else:
                new_sequential.add_module(name, module)
        return new_sequential

    def delete_relu_in_block(block, args_model: dict):
        # features first, should act in-place
        new_features = delete_relu_in_sequential(block.features, args_model)
        block.features = new_features

        # deal with the activation
        new_activation = delete_relu_in_sequential(block.activation, args_model)
        block.activation = new_activation
        

    # modify the head
    new_head = delete_relu_in_sequential(model.head, args_model)
    model.head = new_head

    # deal with blocks, try to obtain all the blocks
    for name, block in model.layer1.named_children():
        delete_relu_in_block(block, args_model)
    for name, block in model.layer2.named_children():
        delete_relu_in_block(block, args_model)
    for name, block in model.layer3.named_children():
        delete_relu_in_block(block, args_model)
    for name, block in model.layer4.named_children():
        delete_relu_in_block(block, args_model)

    return model

def add_relu(model: nn.Module, args_model: dict):

    def add_relu_in_sequential(sequential, args_model: dict, last_conv_name):
        new_sequential = nn.Sequential()
        for name, module in sequential.named_children():
            new_sequential.add_module(name, module)
            if isinstance(module, nn.Conv2d):
                if name != last_conv_name:
                    relu_name = name.replace('conv', 'relu')
                    new_sequential.add_module(relu_name, nn.ReLU(inplace=True))
        return new_sequential

    def add_relu_in_block(block, args_model: dict, last_conv_name):
        # features first, should act in-place
        new_features = add_relu_in_sequential(block.features, args_model, last_conv_name)
        block.features = new_features

        # deal with the activation
        block.activation.add_module('relu', nn.ReLU(inplace=True))
        

    # modify the head
    if isinstance(model, Resnet18):
        last_conv_name = 'conv2'
    elif isinstance(model, Resnet50):
        last_conv_name = 'conv3'
    new_head = add_relu_in_sequential(model.head, args_model, last_conv_name)
    model.head = new_head

    # deal with blocks, try to obtain all the blocks
    for name, block in model.layer1.named_children():
        add_relu_in_block(block, args_model, last_conv_name)
    for name, block in model.layer2.named_children():
        add_relu_in_block(block, args_model, last_conv_name)
    for name, block in model.layer3.named_children():
        add_relu_in_block(block, args_model, last_conv_name)
    for name, block in model.layer4.named_children():
        add_relu_in_block(block, args_model, last_conv_name)

    return model

def add_batchnorm(model: nn.Module, args_model: dict):

    def add_batchnorm_in_sequential(sequential, args_model: dict):
        new_sequential = nn.Sequential()
        for name, module in sequential.named_children():
            new_sequential.add_module(name, module)
            if isinstance(module, nn.Conv2d):
                bn_name = name.replace('conv', 'bn')
                new_sequential.add_module(bn_name, nn.BatchNorm2d(module.out_channels))
        return new_sequential

    def add_batchnorm_in_block(block, args_model: dict):
        # features first, should act in-place
        new_features = add_batchnorm_in_sequential(block.features, args_model)
        block.features = new_features
        

    # modify the head
    new_head = add_batchnorm_in_sequential(model.head, args_model)
    model.head = new_head

    # deal with blocks, try to obtain all the blocks
    for name, block in model.layer1.named_children():
        add_batchnorm_in_block(block, args_model)
    for name, block in model.layer2.named_children():
        add_batchnorm_in_block(block, args_model)
    for name, block in model.layer3.named_children():
        add_batchnorm_in_block(block, args_model)
    for name, block in model.layer4.named_children():
        add_batchnorm_in_block(block, args_model)

    return model

def add_divnorm(model: nn.Module, args_model: dict):

    asact = args_model['divnorm_specs']['asact']

    def add_divnorm_in_sequential(sequential, args_model: dict, last_conv_name):
        if asact:
            new_sequential = nn.Sequential()
            for name, module in sequential.named_children():
                new_sequential.add_module(name, module)
                if isinstance(module, nn.Conv2d):
                    if name != last_conv_name:
                        dn_name = name.replace('conv', 'dn')
                        dn = get_divnorm(module.out_channels, args_model['divnorm_specs'])
                        new_sequential.add_module(dn_name, dn)
        else:
            new_sequential = nn.Sequential()
            for name, module in sequential.named_children():
                new_sequential.add_module(name, module)
                if isinstance(module, nn.Conv2d):
                    dn_name = name.replace('conv', 'dn')
                    dn = get_divnorm(module.out_channels, args_model['divnorm_specs'])
                    new_sequential.add_module(dn_name, dn)
                    
        return new_sequential

    def add_divnorm_in_block(block, args_model: dict, last_conv_name):
        # features first, should act in-place
        new_features = add_divnorm_in_sequential(block.features, args_model, last_conv_name)
        block.features = new_features

        if asact:
            if last_conv_name == 'conv3':
                dn = get_divnorm(block.features.conv3.out_channels, args_model['divnorm_specs']) 
            elif last_conv_name == 'conv2':
                dn = get_divnorm(block.features.conv2.out_channels, args_model['divnorm_specs']) 
            block.activation.add_module('dn', dn)
            
    # modify the head
    if isinstance(model, Resnet18):
        last_conv_name = 'conv2'
    elif isinstance(model, Resnet50):
        last_conv_name = 'conv3'
    new_head = add_divnorm_in_sequential(model.head, args_model, last_conv_name)
    model.head = new_head

    # deal with blocks, try to obtain all the blocks
    for name, block in model.layer1.named_children():
        add_divnorm_in_block(block, args_model, last_conv_name)
    for name, block in model.layer2.named_children():
        add_divnorm_in_block(block, args_model, last_conv_name)
    for name, block in model.layer3.named_children():
        add_divnorm_in_block(block, args_model, last_conv_name)
    for name, block in model.layer4.named_children():
        add_divnorm_in_block(block, args_model, last_conv_name)

    return model

def init_weights(model: nn.Module, args_model: dict) -> nn.Module:

    if args_model['weight_init'] == 'default':
        print('Init weight: default')

    elif args_model['weight_init'] == 'kaiming':
        print('Init weight: kaiming')
        nonlinearity = args_model['nonlinearity']
        for module in model.modules():
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
        print('Init weight: normal')
        for module in model.modules():
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

    elif args_model['weight_init'] == 'multivariate_gaussian':
        p = float(args_model['divnorm_specs']['p'])
        print(f"Init weight: multivariate_gaussian with p = {p}")
        for module in model.modules():
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
        for module in model.modules():
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