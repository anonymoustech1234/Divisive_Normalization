"""
Try to train the model.

******************How to run******************

python3 experiments/train_model.py <exp name> <dataset> <model> *<key1-key2=value pairs>

*************Common Key Value Pairs Example**************

data-bz=128

training-epochs=10

optimizer-name=adam
optimizer-lr=0.01

model-divnorm=ours / model-divnorm=nonorm

lr_scheduler-patience=5

******************Call Example******************

python3 experiments/train_model.py test_1001 imagenet email_result=True optimizer-name=adam optimizer-lr=0.01
python3 experiments/train_model.py test_p_1 imagenet model-divnorm=ours model-divnorm_specs-p=1 
python3 experiments/train_model.py test_new imagenet model-divnorm=nonorm optimizer-lr=0.05
python3 experiments/train_model.py test_new cifar100 alexnet model-divnorm=ours
python3 experiments/train_model.py test_p_1 imagenet resnet50 model-divnorm=ours

"""
import sys
import os
# append the file location to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import multiprocessing

import torch

from modules.utils.read_config import read_config_in_dir
from modules.data.get_dataloaders import get_dataloaders
from modules.network.get_model import get_model
from modules.train.get_trainer import get_trainer, prepare_trainer
from modules.recorder.get_recorder import get_recorder
from modules.seed.set_seed import set_overall_seed

from modules.utils.send_emails import send_emails, format_result

def change_config(args, config_dict):
    """
    args: list of elements in format "key1-key2=0.001". 
        !!! No space around equal sign / before after each value
    config_dict: dictionary of args/configs
    """
    def change_key_value(config_key, value):
        try:# Try converting to int
            out_value = int(value)
        except ValueError:
            try:# Try converting to float
                out_value = float(value)
            except ValueError:
                if value.lower() == 'true':
                    out_value = True
                elif value.lower() == "false":
                    out_value = False
                else:
                    out_value = value # keep str elsewise
        arg_keys = config_key.split('-')
        return arg_keys, out_value

    #Loop through all key_values:
    for key_value in args:
        config_key, value = key_value.split('=')
        config_key, value = change_key_value(config_key, value)
        print(config_key,value)
        if len(config_key) == 1:
            config_dict[config_key[0]] = value
        elif len(config_key) == 2:
            config_dict[config_key[0]][config_key[1]] = value   
        elif len(config_key) == 3:
            config_dict[config_key[0]][config_key[1]][config_key[2]] = value  
    return config_dict

if __name__ == '__main__':

    # SET UP
    if not torch.cuda.is_available():
        raise Exception("CUDA is not detacted on this machine")

    exp_name = sys.argv[1]
    dataset = sys.argv[2]
    model = sys.argv[3]
    args = sys.argv[4:]

    if dataset == 'imagenet':
        # only pytorch dataloader which read from file benefit from this?
        # TODO: test how multiprocessing forkserver work with NVIDIA DALI 
        multiprocessing.set_start_method("forkserver")

    # Read the config
    if dataset == 'imagenet':
        if model == 'alexnet':
            config_dict = read_config_in_dir('configs/', 'config_alexnet_imagenet.json')
        elif model == 'resnet50':
            config_dict = read_config_in_dir('configs/', 'config_resnet50_imagenet.json')
        elif model == 'resnet18':
            config_dict = read_config_in_dir('configs/', 'config_resnet18_imagenet.json')
        elif model == 'vgg16':
            config_dict = read_config_in_dir('configs/', 'config_vgg16_imagenet.json')
    elif dataset == 'cifar100':
        config_dict = read_config_in_dir('configs/', 'config_alexnet_cifar100.json')
    config_dict['exp_name'] = exp_name
    config_dict = change_config(args, config_dict)

    # # # # debug use
    # config_dict['data']['path'] = "/mnt/datasets/CVDatasets/ImageNet/Shub/"
    # config_dict['data']['path'] = "/mnt/datasets/"
    # config_dict['data']['bz'] = 128
    # # # config_dict['data']['dataset'] = 'imagenet'
    # # # config_dict['model']['divnorm'] = "ken"
    # # # config_dict['model']['divnorm_specs']['type'] = 'ken'
    # # # config_dict['model']['divnorm_specs']['fix_all'] = True
    # config_dict['data']['num_workers'] = 10
    # config_dict['training']['epochs'] = 1
    # config_dict['data']['path'] = "/mnt/datasets/CVDatasets/"

    print(config_dict)

    # EXPERIMENTS

    seed = config_dict['seed']
    set_overall_seed(seed)

    loaders = get_dataloaders(config_dict['data'])
    model = get_model(config_dict['model'])
    recorder = get_recorder(config_dict,model)

    args_train, model, optimizer, lr_scheduler, criterion, loaders \
        = prepare_trainer(config_dict, model, loaders)

    trainer = get_trainer(args_train, model, optimizer, lr_scheduler, criterion, loaders,recorder)

    best_val_acc, runtime = trainer.train()

    # RESULTS

    print(f"best_val_acc: {best_val_acc}")
    print(f"run time: {runtime}")

    if config_dict['email_result']:
        # send the email
        # arrange the information
        train_loss, train_acc, val_loss, val_acc = trainer.get_stats()
        email_context = format_result(train_loss, train_acc, val_loss, val_acc, runtime)

        subject = "exp " + config_dict['exp_name'] + " result"
        recipents = config_dict['email_recipents']
        send_emails(subject, email_context, recipents)


