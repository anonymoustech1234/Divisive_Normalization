"""
Functions to create imagenet datasets

1. Create dataset object
2. Create transforms
3. Train_Val split if necessary
"""

import torch
from torch.utils.data import Dataset, ConcatDataset, random_split
from torchvision import transforms, datasets

from typing import Tuple
import os
import sys
# append the file location to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# some packages for downloading imagenet
import requests 
import tqdm
import tarfile
import time

from modules.utils.read_config import read_config_in_dir

# directly use imagefolder class
def get_datasets(args_data: dict) -> Tuple[Dataset, Dataset, Dataset]:
    # return the train/val/test dataset objects
    # is it possible for us to accomodate for different types of imagenets?
    prepare_files(args_data)

    # some configs
    path = args_data['path']

    train_path = os.path.join(path, 'imagenet_train', 'train')
    val_path = os.path.join(path, 'imagenet_val_50k', 'val')

    # OBTAIN TRANSFORMS
    train_transform, val_transform, test_transform = get_transforms(args_data)
            
    # CREATE DATASETS
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=val_transform)

    return train_dataset, val_dataset, None

def get_transforms(args_data: dict):
    # return the train/val/test transforms 
    train_transforms = transforms.Compose([
        # the random resized crop is temp changed into center crop, validate the method
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])

    return train_transforms, val_transforms, test_transforms

def prepare_files(args_data: dict) -> Tuple[Dataset, Dataset, Dataset]:
    # return the train/val/test dataset objects
    # is it possible for us to accomodate for different types of imagenets?

    # some configs
    path = args_data['path']
    imagenet_path = os.path.join(path, 'imagenet.tar')
    imagenet_val_path = os.path.join(path, 'imagenet_val_50k.tar.tar')
    imagenet_url = 'http://s3-west.nrp-nautilus.io/keyu_imagenet/imagenet_train.tar'
    imagenet_url_inside = 'http://rook-ceph-rgw-nautiluss3.rook/keyu_imagenet/imagenet_train.tar'
    imagenet_val_url = 'http://s3-west.nrp-nautilus.io/keyu_imagenet/imagenet_val_50k.tar'
    imagenet_val_url_inside = 'http://rook-ceph-rgw-nautiluss3.rook/keyu_imagenet/imagenet_val_50k.tar'

    # Train Dataset
    # check if the dataset is already ready
    if os.path.exists(os.path.join(path, 'imagenet_train', 'train')):
        imagenet_downloaded = True
    else:
        # check if the file is already downloaded or not
        if os.path.exists(os.path.join(path, 'imagenet.tar')):
            extract_imagenet(imagenet_path, path)
            imagenet_downloaded = True
        else:
            imagenet_downloaded = False

    if not imagenet_downloaded:
        download_imagenet(imagenet_url, imagenet_path)
        extract_imagenet(imagenet_path, path)

    # Val Dataset
    # check if the dataset is already ready
    if os.path.exists(os.path.join(path, 'imagenet_val_50k', 'val')):
        imagenet_val_downloaded = True
    else:
        # check if the file is already downloaded or not
        if os.path.exists(os.path.join(path, 'imagenet_val_50k.tar')):
            extract_imagenet(imagenet_val_path, path)
            imagenet_val_downloaded = True
        else:
            imagenet_val_downloaded = False

    if not imagenet_val_downloaded:
        download_imagenet(imagenet_val_url, imagenet_val_path)
        extract_imagenet(imagenet_val_path, path)


def download_imagenet(url, file_path, retries=15, backoff_factor=1):
    directory = os.path.dirname(file_path)

    # create the directory if it does not exist
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    for attempt in range(retries):
        try:
            print("Attempt", attempt + 1, "to download ImageNet dataset...")

            # Checking for partial download
            if os.path.exists(file_path):
                resume_byte_pos = os.path.getsize(file_path)
                headers = {'Range': f'bytes={resume_byte_pos}-'}
                print("Resuming download...")
            else:
                resume_byte_pos = 0
                headers = {}

            res = requests.get(url, headers=headers, stream=True)
            res.raise_for_status()  # Ensure the request was successful

            # Determine the file size for the progress bar
            total_file_size = int(res.headers.get('Content-Range').split('/')[1]) if resume_byte_pos else int(res.headers.get('Content-Length', 0))
            file_size = total_file_size - resume_byte_pos

            # Open the file in append mode if partially downloaded
            mode = 'ab' if resume_byte_pos else 'wb'
            with open(file_path, mode) as file, tqdm.tqdm(
                    total=file_size, initial=resume_byte_pos,
                    unit='B', unit_scale=True, unit_divisor=1024
                ) as progress_bar:
                
                block_size = 4096
                for chunk in res.iter_content(chunk_size=block_size):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))

            # Verify complete download
            if os.path.getsize(file_path) != total_file_size:
                raise Exception("Download incomplete. File size does not match expected size.")
            print("Download completed successfully")
            return True
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
            if "Download incomplete" in str(e):
                print("Download incomplete. Attempting to retry...")
            else:
                raise e  # Re-raise unexpected exceptions
        print("Waiting to retry...")
        print("Waiting to retry...")
        time.sleep(backoff_factor * (2 ** attempt))

    # After all retries, if download was not successful, consider cleaning up or notifying the user
    print("Failed to download after all retries.")
    # Optionally delete partial file or take other action
    return False

def extract_imagenet(file_path: str, extract_path: str) -> None:
    # extract the imagenet files from tar
    if not os.path.exists(file_path):
        raise FileNotFoundError("Imagenet is not found.")
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    with tarfile.open(file_path) as tar:
        print("Extracting imagenet_s tar file")
        tar.extractall(path=extract_path)

if __name__ == "__main__":
    # if try to directly run this file
    # will try to download & extract imagenet directly

    config_dict = read_config_in_dir('configs', 'config_alexnet_imagenet.json')
    get_datasets(config_dict['data'])
