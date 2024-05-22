"""
This file provides a simple way to test model on datasets, and is designed to run locally.

This file will solely depending on a test config, called config_test.json.
In config_test.json, the path toward
- model to be tested
- datasets to be tested
are defined

The config should also provide the exact config for the model to be tested.
The config of dataset should in such structure

- Datasets
    - Datasets to be tested 1
        - Class 1
        - Class 2
        ...
    - Datasets to be tested 2
        ...
    ...

The function then test each model on each dataset. Saves as pandas CSV.

******************How to run******************

python3 experiments/test_model.py *<key1-key2=value pairs>

"""
import sys
import os
# append the file location to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import multiprocessing
import copy
import pandas as pd

import torch

from modules.utils.read_config import read_config_in_dir
from modules.data.get_dataloaders import get_dataloaders
from modules.network.get_model import load_model
from modules.test.get_tester import get_tester, prepare_tester
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

def get_folders(root_path):
    paths_dict = {}

    for folder_2 in os.listdir(root_path):
        path_level_2 = os.path.join(root_path, folder_2)
        key = folder_2
        paths_dict[key] = path_level_2

    return paths_dict

if __name__ == '__main__':

    args = sys.argv[1:]

    # Read the config
    config_dict = read_config_in_dir('configs/', 'config_test.json')
    config_dict = change_config(args, config_dict)

    # # Debug Use
    # config_dict['data']['path'] = "/mnt/datasets/"

    # # Do we really have any randomness in test model?
    # seed = config_dict['seed']
    # set_overall_seed(seed)

    # TODO: use a for loop to extract all sub_directory from the root data structure
    path_dicts = get_folders(config_dict['data']['path'])
    results = {}
    for key in path_dicts:
        new_config = copy.deepcopy(config_dict)
        new_config['data']['path'] = path_dicts[key]

        _, _, test_loader = get_dataloaders(new_config['data'])
        model = load_model(config_dict['model'])

        args_testing, model, criterion, loader = prepare_tester(config_dict, model, test_loader)
        tester = get_tester(args_testing, model, criterion, loader)

        test_loss, test_acc, test_acc_top5 = tester.test()

        results[key] = [test_loss, test_acc, test_acc_top5]
        
        del tester, model, test_loader

    # TODO: Save the result into a csv using pandas like result
    results_df = pd.DataFrame.from_dict(results, orient='index').sort_index().rename(columns={0: 'loss', 1: 'acc_top_1', 2: 'acc_top_5'})
    results_df.to_csv(os.path.join(config['testing']['result_path']))
    
