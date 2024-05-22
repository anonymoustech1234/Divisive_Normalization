from modules.network.models.alexnet import Alexnet
from modules.network.models.resnet50 import Resnet50
from modules.network.models.resnet18 import Resnet18
from modules.network.models.vgg16 import VGG16

import torch
import torch.nn as nn

def get_model(args_model: dict) -> nn.Module:
    """
    Return the network ready for training & testing.
    """

    print("Preparing Model")

    # obtain the basic model structure
    if args_model['type'] == 'alexnet':
        model = Alexnet(args_model)
        import modules.network.alexnet_modifier as net_modifer
    elif args_model['type'] == 'resnet50':
        model = Resnet50(args_model)
        import modules.network.resnet_modifier as net_modifer
    elif args_model['type'] == 'resnet18':
        model = Resnet18(args_model)
        import modules.network.resnet_modifier as net_modifer
    elif args_model['type'] == 'vgg16':
        model = VGG16(args_model)
        import modules.network.vgg_modifier as net_modifer
    else:
        raise NotImplementedError(f"Model {args_model['type']} not implemented")

    # modify the relu/divnorm/batchnorm, structures
    # since we want the structure to be conv + relu + divnorm + batchnorm,
    # and those methods add layers right after relu, add them reversely
    model = net_modifer.delete_relu(model, args_model)

    if args_model['batchnorm']:
        model = net_modifer.add_batchnorm(model, args_model)

    if args_model['divnorm_specs']['type'] == 'ours' or args_model['divnorm_specs']['type'] == 'ken':
        # add either our or ken's divnorm
        model = net_modifer.add_divnorm(model, args_model)
    
    if args_model['relu']:
        model = net_modifer.add_relu(model, args_model)

    # initialize the weight
    model = net_modifer.init_weights(model, args_model)

    print(model)
    
    return model

def load_model(args_model: dict) -> nn.Module:
    """
    TODO: Understand what needs to be load.
    """
    print("Loading Model")

    # obtain the basic model structure
    if args_model['type'] == 'alexnet':
        model = Alexnet(args_model)
        import modules.network.alexnet_modifier as net_modifer
    elif args_model['type'] == 'resnet50':
        model = Resnet50(args_model)
        import modules.network.resnet_modifier as net_modifer
    elif args_model['type'] == 'resnet18':
        model = Resnet18(args_model)
        import modules.network.resnet_modifier as net_modifer
    elif args_model['type'] == 'vgg16':
        model = VGG16(args_model)
        import modules.network.vgg_modifier as net_modifer
    else:
        raise NotImplementedError(f"Model {args_model['type']} not implemented")

    pass
    