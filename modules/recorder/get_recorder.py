"""
Experiment Recorder

Functions
1. record model configs, train, validation loss and accuracy per epoch to txt + end to txt
2. Transform config, training log txt into json (end)
2. record neuron activity rate (flag on/off) (per epoch + end)
3. record divnorm parameters over epochs (per epoch + end)
4. record layer weights (initial, best model, end model) (initial, end)

Beaware:
1. consider the situation if train not full epoch
2. refers to the model structure, parameter access, paramter dimension
3. ...

Expected output file
config dict json file with configs, acc, loss
conv1 weight txt file with initial, best, final
conv1 neuron activity csv files for conv1: conv1, relu1, dn1, bn1
divnorm1 parameters json file with sigma, gamma

optional logging file
acc, losslog



"""

LOG_PATH = './run'


import numpy as np
import pandas as pd
import os
import json
import ast
import matplotlib.pyplot as plt

import torch
from copy import deepcopy

class experiment_recorder():
    def __init__(self,configs, model)-> None:
        """
        Takes in: model object, configuration dictionary
        """
        
        self.model = model
        self.dataset_name = configs['data']['dataset']
        self.exp_name = configs['exp_name']

        #flags:
        #TODO: update config file to include flag parameters, change below to read in config
        self.model_saving_flag = configs['recorder']['model_saving_flag']
        self.neuron_activity_flag = False 
        self.train_log = False #if true, ini a txt, update at each epoch for acc, loss, divnormpara

        #paths
        self.record_folder_path = os.path.join(LOG_PATH, configs['exp_name'])
        check_path(self.record_folder_path)
        self.record_file_path = os.path.join(self.record_folder_path, f"{configs['exp_name']}.txt") 
        self.record_json_path = os.path.join(self.record_folder_path, f"{configs['exp_name']}.json") 
        self.record_model_path = os.path.join(self.record_folder_path, f"{configs['exp_name']}_best_model.pth")

        #recorded features initialization
        self.current_epoch = 0
        self.destination_epoch = configs['training']['epochs']
        self.status_dict = {'configs':configs, 
                            'device':get_cuda_device_name(),
                            'best_acc':0, 'best_loss':100000, 'end_epoch':self.current_epoch,
                            'train_acc':[],'train_loss':[],
                            'val_acc':[],'val_loss':[]
                            }
        self.divnorm_dict  = {'gamma':{},'sigma':{}}

        #record model status dict if saving flag on
        self.model_dict_log = None
        if self.model_saving_flag:
            self.best_model_dict_log=deepcopy(model.state_dict())
        
        self.conv1_weights_dict = {'initial':model.get_layers()['conv1'].weight.data.detach().cpu().numpy(), 
                                   'best':model.get_layers()['conv1'].weight.data.detach().cpu().numpy(), 
                                   'final':None}#use model function: model.get_weight(self) --> dict of model layer objects
        
        
        
        # np.savetxt(self.record_folder_path+'conv1_weights.txt', 
        #            self.conv1_weights_dict['initial'].flatten(), 
        #            header='initial', comments='')

        log_to_file(self.record_file_path, '\n' + str(configs))

        #TODO: create files , folders for data storage
        write_to_json(self.record_json_path, self.status_dict) #config dict file
        #weight file
        
        #train log
        if self.train_log:
            ... #create acc, loss log file
            ... #divnorm log file
            
        print(f'results will be stored in {self.record_folder_path}')

        return None
    
    def all_recorder_per_epoch(self, model,train_loss, train_acc, val_loss, val_acc):
        #TODO: run all the recorders ( ... ) based on given flags for each epcoh
        """
        This function should be called inbetween epochs
        self.current_epoch+1
        update self variables
        append to file logs
        """
        #conv1_weights = self.model.getweights
        #self.calculate_neuron_activity()

        #update acc, loss
        self.status_dict['train_acc'].append(train_acc)
        self.status_dict['train_loss'].append(train_loss)
        self.status_dict['val_acc'].append(val_acc)
        self.status_dict['val_loss'].append(val_loss)

        #update temporal log
        self.current_epoch +=1
        ...

        #update divnorm para
        self.divnorm_dict  = {'gamma':{},'sigma':{}}

        #self.calculate_neuron_activity() 

    def update_best_result(self, best_model, train_loss, train_acc, val_loss, val_acc):
        """call recorder functions to update the best models"""
        #self.log_best_model(self, best_model)
        self.status_dict['best_acc'] = val_acc
        self.status_dict['best_loss'] = val_loss
        self.status_dict['best_train_acc'] = train_acc
        self.status_dict['best_train_loss'] = train_loss
        self.conv1_weights_dict['best']=best_model.get_layers()['conv1'].weight.data.detach().cpu().numpy()
        if self.model_saving_flag:
            self.best_model_dict_log = deepcopy(best_model.state_dict())
        #TODO: IO costly addition: directly update the json file with best acc/loss when this function is called

            
                            
    def all_recorder_final(self, current_model):
        """
        This function should be called at the end of the trail
        check if current epoch == final epoch
        document self.variales into files
        delete unnecc log files
        """
        self.model = current_model
        self.status_dict['end_epoch']=self.current_epoch,
        #TODO: run all the recorders ( ... ) based on given flags
        #model stats
        import torch

        # Check tensor in dict
        def convert_tensor(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to('cpu').tolist()  # convert tensor to list
            elif isinstance(obj, dict):
                return {k: convert_tensor(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensor(v) for v in obj]
            else:
                return obj
        serializable_data = convert_tensor(self.status_dict)

        write_to_json_in_dir(self.record_folder_path,f'{self.exp_name}.json',serializable_data)

        #model dict
        if self.model_saving_flag:
            ...
            #torch.save(self.best_model_dict_log, 'best_model.pth')
            model_dict = self.best_model_dict_log
            state_dict = {'model': model_dict} #'optimizer': self.__optimizer.state_dict()
            torch.save(state_dict, self.record_model_path)
            print(f'best model savef to {self.record_model_path}')

        #weights
        self.conv1_weights_dict['final'] = current_model.get_layers()['conv1'].weight.data.detach().cpu().numpy()
        for status, weights in self.conv1_weights_dict.items():
            self.log_weights(weights, model_status=status)      
        ...

    def update_recorder_status_dict(self, key, value):
        """only support outside edit to first level keys"""
        self.status_dict[key] = value

    def append_recorder_status_dict(self, key, value):
        """
        if key doesn't exit, add key and value as list
        if key exist, append value
        """
        if key in self.status_dict:
            self.status_dict[key].append(value)
        else:
            self.status_dict[key] =[value]

        


    def calculate_neuron_activity(self,current_model):
        ...
    
    # def log_best_model(self, model):
    #     best_model_dict = deepcopy(model.state_dict())
    #     self.log_best_model_dict(best_model_dict)
        
    # def log_best_model_dict(self, best_model_dict):
    #     """
    #     call at end of each epoch in get_trainer.py
    #     usecase: takes in trainer.__best_model ( = deepcopy(model.state_dict()))
    #     """
    #     self.best_model_dict_log = best_model_dict

    def log_best_weight(self, weights):
        """
        call at end of each epoch in get_trainer.py
        usecase: takes in trainer.__best_model ( = deepcopy(model.state_dict()))
        """
        self.self.conv1_weights_dict['best'] = weights
    
    def update_flag(self, divnorm):
        if divnorm =='ours':
            ...
        elif divnorm == 'ken':
            #false gamma and variance flag regardless of config
            ...
        else:
            #assume nonorm
            #set gamma, sigma, variance flag to false regardless of config
            ...
        return 
    def log_weights(self, weights, model_status='initial'):
        #TODO: log weights to given path
        """
        log stored neuron activity and divnomr parameters to csv / txt files
        """
        #Log conv1 weight:
        #
        weight_log_path = os.path.join(self.record_folder_path,f'{self.exp_name}_conv1_weights.txt')
        with open(weight_log_path, 'ab') as f:
            np.savetxt(f, weights.flatten(), header=model_status, comments='')
        print(f'{model_status} weight saved to {weight_log_path}')

        #TODO: do we want to record other layers weight? (since when we save model we only get the best model, do we want to reocrd a initial stage?)

        '''
        #log initial, best and final filters of the model conv1 layer
        # best_weight  = self.__best_model['conv1.weight'].data.detach().cpu().numpy()
        # final_weight = self.__model.conv1.weight.data.detach().cpu().numpy()
        # with open(self.__log_path+'conv1_weights.txt', 'ab') as f:
        #     np.savetxt(f, best_weight.flatten(), header='best', comments='')
        #     np.savetxt(f, final_weight.flatten(), header='end', comments='')
        # print(f'weight saved to {self.__log_path}conv1_weights.txt')
        '''
        return 
    
    def log_neuron_activity(self):
        #log the neuron activity #AAAAA
        # conv1_var = self.__conv1_variance.numpy()
        # relu1_var = self.__relu1_variance.numpy()
        # dn1_var = self.__dn1_variance.numpy()
        # bn1_var = self.__bn1_variance.numpy()
        # np.savetxt(self.__log_path+'conv1_var.csv', conv1_var.reshape(dn1_var.shape[0], -1), delimiter=',')
        # np.savetxt(self.__log_path+'relu1_var.csv', relu1_var.reshape(dn1_var.shape[0], -1), delimiter=',')
        # np.savetxt(self.__log_path+'dn1_var.csv', dn1_var.reshape(dn1_var.shape[0], -1), delimiter=',')
        # np.savetxt(self.__log_path+'bn1_var.csv', bn1_var.reshape(bn1_var.shape[0], -1), delimiter=',')
        print(f'Neuron Activity saved to {self.__log_path}conv1_var.csv etc')


    def set_recorder(self):
        #change recorder configs
        #TODO: decude what will be changed during experiment
        return None
    
    def set_model(self, current_model):
        self.model = current_model

    def get_status_dict(self):
        return self.status_dict

def get_recorder(configs, model):
    """
    create a return an experiment_recorder object
    """
    return experiment_recorder(configs,model)   

# ########### Util Functions #############

#File reading / loading functions
def check_path(path):
    """
    Check if a path exists, if not, create it.
    Args:
        path (str): The path to check or create.
    Returns:
        bool: True if the path exists or was successfully created, False otherwise.
    """
    if os.path.exists(path):
        return True
    else:
        try:
            os.makedirs(path) 
            return True
        except Exception as e:
            print(f"Error creating path {path}: {e}")
            return False

def read_json(path):
    """read file from given path"""
    if os.path.isfile(path):
        with open(path) as json_file:
            data = json.load(json_file)
        return data
    else:
        raise Exception("file doesn't exist: ", path)
    
def read_json_in_dir(root_dir, file_name):
    path = os.path.join(root_dir, file_name)
    return read_json(path)


def write_to_json(path, data):
    with open(path, "w") as outfile:
        json.dump(data, outfile)

    #possible update:
    # if isinstance(data, dict):
    #     with open(path, "w") as outfile:
    #         json.dump(data, outfile)
    # elif isinstance(data, str):
    #     with open(path, "w") as outfile:
    #         outfile.write(data)

def write_to_json_in_dir(root_dir, file_name, data):
    path = os.path.join(root_dir, file_name)
    write_to_json(path, data)

def log_to_file(path, log_str):
    with open(path, 'a') as f:
        f.write(log_str + '\n') #logged after the original content

def log_to_file_in_dir(root_dir, file_name, log_str):
    path = os.path.join(root_dir, file_name)
    log_to_file(path, log_str)


def get_cuda_device_name():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        #print("GPU Name:", torch.cuda.get_device_name(device))
        return torch.cuda.get_device_name(device)
    else:
        #print("CUDA is not available. Running on CPU.")
        return 'cpu'