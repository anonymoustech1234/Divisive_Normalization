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

def get_tester(args_testing, model, criterion, loader):
    return Tester(args_testing, model, criterion, loader)

def prepare_tester(args: dict, model: torch.nn.Module, loader):
    # Prepare what's needed for creating a trainer

    criterion = get_criterion(args['testing'])

    return args['testing'], \
        model, \
        criterion, \
        loader

class Tester(object):
    """
    Trainer class is designed to be SPECIFIC for one whole training process
    of one model.
    """

    def __init__(self, args_testing, model, criterion, loader):

        # init parameters
        self.__model = model
        self.__criterion = criterion
        self.__test_loader = loader

        self.__test_loss = None
        self.__test_acc = None
        self.__test_acc_top5 = None

        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__device_name = torch.cuda.get_device_name(torch.cuda.current_device()) \
            if torch.cuda.is_available() else 'cpu'
        print("This experiment is running on device:", self.__device_name)

        # move to the device
        self.__model = self.__model.to(self.__device).float()
        self.__criterion = self.__criterion.to(self.__device).float()

    def test(self):

        self.__model.eval()
        loader = self.__test_loader

        test_loss = 0.0
        test_correct = 0.0
        test_correct_k = 0.0

        total_number_images = len(loader.dataset)
        total_number_batches = len(loader)

        with torch.no_grad():
            for data in loader:

                images, labels = data
                images, labels = images.to(self.__device), labels.to(self.__device)

                logits = self.__model(images)
                loss, correct = self.__compute_loss_accuracy(logits, labels)
                correct_k = self.__compute_top_k_accuracy(logits, labels, topk = 5)

                test_loss += loss.detach().cpu()
                test_correct += correct
                test_correct_k += correct_k

        self.__test_loss = test_loss.detach().cpu().item() / total_number_batches
        self.__test_acc = test_correct * 100 / total_number_images
        self.__test_acc_top5 = test_correct_k * 100 / total_number_images

        return self.__test_loss, self.__test_acc, self.__test_acc_top5


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
    
    def get_criterion(self):
        return self.__criterion
    
    def get_loader(self):
        return self.__test_loader
    
    def get_device(self):
        return self.__device
    
    def get_stats(self):
        return self.__test_loss, self.__test_acc, self.__test_acc_top5
    
