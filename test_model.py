from modules.recorder.get_recorder import get_recorder
from modules.network.get_model import get_model
from modules.utils.read_config import read_config_in_dir
from experiments.train_model import *

import sys
import os
# append the file location to the path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import multiprocessing

from modules.utils.read_config import read_config_in_dir
from modules.data.get_dataloaders import get_dataloaders
from modules.network.get_model import get_model
from modules.train.get_trainer import get_trainer, prepare_trainer
from modules.recorder.get_recorder import get_recorder

from modules.utils.send_emails import send_emails, format_result

model_config = {'type': 'alexnet',
 'mean_std': 0.01,
 'std': 0.01,
 'std_ratio': 0.01,
 'batchnorm': True,
 'activation': 'half-square',
 'weight_init': 'kaiming',
 'l1reg': 0,
 'relu': True,
 'for_dataset': 'imagenet',
 'num_classes': 1000,
 'divnorm_specs': {'type': 'nonorm',
  'fix_all': False,
  'single_gamma': False,
  'single_sigma': False,
  'fix_gamma': False,
  'fix_sigma': False,
  'normalize_gamma': False,
  'normalize_sigma': False,
  'eps': 0,
  'sigma': 1.0,
  'gamma': 1.0,
  'lamb': 10.0,
  'alpha': 0.1,
  'beta': 1.0,
  'k': 10.0,
  'p': 0.5,
  'power': 2}}




def get_HCdivmodel():
    model_config['divnorm_specs']['type'] = 'ours'
    print(model_config)
    HCdiv_model = get_model(model_config)
    return HCdiv_model
    

def get_baselinemodel():
    model_config['divnorm_specs']['type'] = 'nonorm'
    print(model_config)
    baseline_model = get_model(model_config)
    return baseline_model
    



