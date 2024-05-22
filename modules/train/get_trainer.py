"""
Executing the model training part

Take in the model, dataloder, optimizer, lr_scheduler, loss
Train the model for certain epochs

Implement the early stopping technique?
"""
import torch
from copy import deepcopy
from tqdm import tqdm
import math
import numpy as np
import time

# Used to distinguish dataloaders type
from torch.utils.data import DataLoader 

from modules.optimizer.get_optimizer import get_optimizer
from modules.lr_scheduler.get_lr_scheduler import get_lr_scheduler
from modules.criterion.get_criterion import get_criterion

def get_trainer(args_training, model, optimizer, lr_scheduler, criterion, loaders, recorder):
    return Trainer(args_training, model, optimizer, lr_scheduler, criterion, loaders, recorder)

def prepare_trainer(args: dict, model: torch.nn.Module, loaders):
    # Prepare what's needed for creating a trainer

    optimizer = get_optimizer(args['optimizer'], model) # this is created before model.to(device)
    lr_scheduler = get_lr_scheduler(args['lr_scheduler'], optimizer)
    criterion = get_criterion(args['training'])

    return args['training'], \
        model, \
        optimizer, \
        lr_scheduler, \
        criterion, \
        loaders

class Trainer(object):
    """
    Trainer class is designed to be SPECIFIC for one whole training process
    of one model.
    """

    def __init__(self, args_training, model, optimizer, lr_scheduler, criterion, loaders, recorder):

        # init parameters
        self.__best_model = deepcopy(model.state_dict())
        self.__model = model
        self.__optimizer = optimizer
        self.__lr_scheduler = lr_scheduler
        self.__criterion = criterion
        self.__train_loader = loaders[0]
        self.__val_loader = loaders[1]
        self.__test_loader = loaders[2]
        self.__recorder = recorder

        self.__total_epochs = args_training['epochs']
        self.__early_stop = args_training['early_stop']
        self.__early_stop_patience = args_training['patience']
        self.__current_epoch = 0
        self.__current_steps = 0

        self.__train_losses = []
        self.__train_accs = []
        self.__train_accs_top5 = []
        self.__val_losses = []
        self.__val_accs = []
        self.__val_accs_top5 = []
        self.__best_val_acc = 0.0

        self.__dynamic_train_loss_lst = []
        self.__dynamic_train_acc_lst = []

        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__device_name = torch.cuda.get_device_name(torch.cuda.current_device()) \
            if torch.cuda.is_available() else 'cpu'
        print("This experiment is running on device:", self.__device_name)

        # move to the device
        self.__model = self.__model.to(self.__device).float()
        self.__criterion = self.__criterion.to(self.__device).float()

        if not isinstance(self.__train_loader, DataLoader):
            # place the import here to avoid environment issue
            from nvidia.dali.plugin.pytorch import DALIGenericIterator
            self.__dataloader_type = 'gpu'
        else:
            self.__dataloader_type = 'cpu'

    def train(self):
        """
        Train to the total epochs. 
        Recorder helps record needed statistics. (include GPU + run time)
        
        return:
            best_val_acc
            run_time
        
        """

        self.__epoch_describer = tqdm(range(self.__current_epoch, self.__total_epochs), desc='Step: 0')
        patience_count = 0
        start_time = time.time()

        #self.__recorder.update_recorder_status_dict('device_name', self.__device_name)

        for epoch in self.__epoch_describer:

            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.test(validate = True)
            self.__recorder.all_recorder_per_epoch(self.__model,train_loss, train_acc, val_loss, val_acc)
            # All of the above are already stored as class variable

            # STOP AGAINST NAN
            if train_loss == -1 and train_acc == -1:
                break 

            # BEST MODEL & EARLY STOPPING
            if val_acc > self.__best_val_acc:
                self.__best_val_acc = val_acc
                self.__best_model = deepcopy(self.__model.state_dict())
                self.__recorder.update_best_result(self.__model, train_loss, train_acc, val_loss, val_acc)

            if self.__early_stop:
                if epoch > 0 and val_loss > self.__val_losses[epoch - 1]:
                    patience_count += 1
                else:
                    patience_count = 0
                if patience_count >= self.__early_stop_patience:
                    print('early stopping')
                    break

            # lr_scheduler
            if self.__lr_scheduler is not None:
                if isinstance(self.__lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.__lr_scheduler.step(val_loss)
                else:
                    self.__lr_scheduler.step()


            # RECORDER FUNCTIONS
                
        end_time = time.time()
        run_time = round(end_time - start_time, 2)
        self.__recorder.update_recorder_status_dict('runtime', run_time)
        self.__recorder.update_recorder_status_dict('top5train', self.__train_accs_top5)
        self.__recorder.update_recorder_status_dict('top5val', self.__val_accs_top5)
        self.__recorder.all_recorder_final(self.__model)
        self.__model.load_state_dict(self.__best_model)
                
        return self.__best_val_acc, run_time


    def train_one_epoch(self):
        
        self.__model.train()
        train_loss = 0.0
        train_correct = 0.0
        train_correct_k = 0.0

        if self.__dataloader_type == 'cpu':
            total_number_images = len(self.__train_loader.dataset)
            total_number_batches = len(self.__train_loader)

        elif self.__dataloader_type == 'gpu':
            total_number_images = self.__train_loader._size
            total_number_batches = 0 # this needs to calculate
        
        for data in self.__train_loader:

            if self.__dataloader_type == 'cpu':
                images, labels = data
                images, labels = images.to(self.__device), labels.to(self.__device)

            elif self.__dataloader_type == 'gpu':
                images = data[0]['images']
                labels = data[0]['labels'].squeeze(-1).to(dtype=torch.long)
                total_number_batches += 1
            
            logits = self.__model(images)
            loss, correct = self.__compute_loss_accuracy(logits, labels)
            correct_k = self.__compute_top_k_accuracy(logits, labels, topk=5)

            train_loss += loss.detach().cpu()
            train_correct += correct
            train_correct_k += correct_k

            # live acc & loss
            self.__dynamic_train_acc_lst.append(correct * 100 / images.shape[0])
            if len(self.__dynamic_train_acc_lst) > 100:
                self.__dynamic_train_acc_lst.pop(0)
            self.__dynamic_train_loss_lst.append(loss.detach().cpu().item())
            if len(self.__dynamic_train_loss_lst) > 100:
                self.__dynamic_train_loss_lst.pop(0)

            if self.__current_steps % 500 == 0:
                self.__epoch_describer.set_description(
                    f"Step: {self.__current_steps}(loss={np.mean(self.__dynamic_train_loss_lst):.2f}, acc={np.mean(self.__dynamic_train_acc_lst):.2f})"
                )
                
            # back prop
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()

            # # release memory
            # del images, labels, loss, correct

            self.__current_steps += 1
        
        self.__current_epoch += 1

        if torch.isnan(train_loss):
            print("NaN loss detected, existing")
            return -1, -1

        self.__train_losses.append(train_loss.detach().cpu().item() / total_number_batches)
        self.__train_accs.append(train_correct * 100 / total_number_images)
        self.__train_accs_top5.append(train_correct_k * 100 / total_number_images)

        return self.__train_losses[-1], self.__train_accs[-1]

    def test(self, validate=True):

        self.__model.eval()
        loader = self.__val_loader if validate else self.__test_loader

        test_loss = 0.0
        test_correct = 0.0
        test_correct_k = 0.0

        if self.__dataloader_type == 'cpu':
            total_number_images = len(loader.dataset)
            total_number_batches = len(loader)

        elif self.__dataloader_type == 'gpu':
            total_number_images = loader._size
            total_number_batches = 0 # this needs to calculate

        with torch.no_grad():
            for data in loader:
                if self.__dataloader_type == 'cpu':
                    images, labels = data
                    images, labels = images.to(self.__device), labels.to(self.__device)

                elif self.__dataloader_type == 'gpu':
                    images = data[0]['images']
                    labels = data[0]['labels'].squeeze(-1).to(dtype=torch.long)
                    total_number_batches += 1

                logits = self.__model(images)
                loss, correct = self.__compute_loss_accuracy(logits, labels)
                correct_k = self.__compute_top_k_accuracy(logits, labels, topk = 5)

                test_loss += loss.detach().cpu()
                test_correct += correct
                test_correct_k += correct_k

        final_loss = test_loss.detach().cpu().item() / total_number_batches
        final_acc = test_correct * 100 / total_number_images
        final_acc_topk = test_correct_k * 100 / total_number_images

        if validate:
            self.__val_losses.append(final_loss)
            self.__val_accs.append(final_acc)
            self.__val_accs_top5.append(final_acc_topk)
        
        return final_loss, final_acc


    def __compute_loss_accuracy(self, logits, labels):
        loss = self.__criterion(logits, labels)
        pred = logits.argmax(dim=1, keepdim=True)
        correct = pred.eq(labels.view_as(pred)).sum().item()
        return loss, correct
        
    def __compute_top_k_accuracy(self, logits, labels, topk=5):

        _, pred = logits.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        correct_k = correct[:topk].reshape(-1).float().sum(0, keepdim=True).item()
        return correct_k

    # Some IO functions
    
    def get_model(self):
        return self.__model

    def get_optimizer(self):
        return self.__optimizer
    
    def get_lr_scheduler(self):
        return self.__lr_scheduler
    
    def get_criterion(self):
        return self.__criterion
    
    def get_loaders(self):
        return self.__train_loader, self.__val_loader, self.__test_loader
    
    def get_device(self):
        return self.__device
    
    def get_stats(self):
        return self.__train_losses, self.__train_accs, self.__val_losses, self.__val_accs
    
