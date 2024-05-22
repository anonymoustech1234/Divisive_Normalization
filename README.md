# Divisive_Normalization

This repository contains source code from the paper *Modeling Divisive Normalization in Visual Cortex*.  
If not intended to use our training pipeline / model, directly use files under `structures` to obtain our implementations of divisive normalization layer.

## Table of Contents
[How to run](#how-to-run)   
[Preferred hardware Settings](#hardware-settings)  
[Environment](#environment)  
[Directory and file structure](#directory-and-file-structure)

## How to run
```shell
$ python3 experiments/train_model.py <dataset name> <model name> <exp name> <key-value pairs>
```

1. The function will automatically download the dataset, be careful.
2. There are two ways of modifying hyperparameters: on command line or on corresponding json file.

### Specific Command Line
- For training Alexnet on CIFAR 100
  ```shell
  $ python3 experiments/train_model.py cifar100 alexnet
  ```
- For training Alexnet on ImageNet
  ```shell
  $ python3 experiments/train_model.py imagenet alexnet
  ```
- Model can be changed into `resnet18`, `resnet50`, `vgg16`

### Controlling Model Structure
There are few hyperparameters in the config used to modify model structure.
1. The existence of **relu**: `model-relu`
2. The existence of **batchnorm**: `model-batchnorm`
3. The existence of **divnorm**: `model-divnorm_specs-type` (`ours` means divnorm, `nonorm` means no divnorm)  
   For divnorm in **resnet**, if wish to use divnorm as activation function, modify `model-divnorm_specs-asact`
#### Example Usage
```shell
$ python3 experiments/train_model.py imagenet alexnet model-relu=false model-divnorm_specs-type=ours
```
This trains alexnet on imagenet with no relu and divisive normalization as the activation function.

## Hardware settings
**For ImageNet**
- ~ 300 G ephemeral storage
- 1 NVIDIA GPU
### CPU Dataloaders
- 12 CPU / 10 num_workers
- 40 G Memory
#### CIFAR100
- 8 CPU / 8 num_workers
- request 6 G / limit 18 G memory
### GPU Dataloaders (preferred & faster for small model)
- 8 CPU / 8 num_workers
- request 12 G / limit 18 G memory


## Environment
Use the Dockerfile in this repo.


## Directory and file structure
- `configs`: contains hyperparameters for executing the experiment
- `experiments`: directly executable python file for experiments / tests
- `modules`: important modules to perform experiments
- `structures`: this project's core structure: implementation of divisive normalization 
