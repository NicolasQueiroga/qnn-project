import os
import shutil
import copy

import torch
from torch.utils.data import Dataset
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def split_images(images_dir, train_dir, test_dir, output_dir, train_ratio):
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir) 
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    os.makedirs(train_dir, exist_ok = True)
    os.makedirs(test_dir, exist_ok = True)
    os.makedirs(output_dir, exist_ok = True)

    classes = os.listdir(images_dir)

    for c in classes:
        class_dir = os.path.join(images_dir, c)
        images = os.listdir(class_dir)
        
        n_train = int(len(images) * train_ratio)
        train_images = images[:n_train]
        test_images = images[n_train:]
        
        os.makedirs(os.path.join(train_dir, c), exist_ok = True)
        os.makedirs(os.path.join(test_dir, c), exist_ok = True)
        
        for image in train_images:
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(train_dir, c, image) 
            shutil.copyfile(image_src, image_dst)
            
        for image in test_images:
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(test_dir, c, image) 
            shutil.copyfile(image_src, image_dst)


def pretrained_info(train_dir):
    train_data = datasets.ImageFolder(root = train_dir, transform = transforms.ToTensor())

    means = torch.zeros(3)
    stds = torch.zeros(3)

    for img, _ in train_data:
        means += torch.mean(img, dim = (1,2))
        stds += torch.std(img, dim = (1,2))

    means /= len(train_data)
    stds /= len(train_data)

    return means, stds


def train_transforms(img_size, means, stds):
    return  transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.RandomRotation(5),
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.RandomCrop(img_size, padding = 10),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = means, 
                                            std = stds)
                    ])
    

def test_transforms(img_size, means, stds):
    return transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.CenterCrop(img_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = means, 
                                            std = stds)
                    ])
    

def get_dataset(images_dir, train_dir, test_dir, output_dir, img_size=256, train_ratio=.8, val_ratio=.9):
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        split_images(images_dir, train_dir, test_dir, output_dir, train_ratio)

    means, stds = pretrained_info(train_dir)

    train_transform = train_transforms(img_size, means, stds)
    test_transform = test_transforms(img_size, means, stds)

    train_data = datasets.ImageFolder(root = train_dir, 
                                  transform = train_transform)

    test_data = datasets.ImageFolder(root = test_dir, 
                                    transform = test_transform)
    
    n_train_examples = int(len(train_data) * val_ratio)
    n_valid_examples = len(train_data) - n_train_examples

    train_data, valid_data = data.random_split(train_data, 
                                            [n_train_examples, n_valid_examples])
    
    valid_data = copy.deepcopy(valid_data)
    valid_data.dataset.transform = test_transforms
    
    return train_data, valid_data, test_data

def get_dataloaders(train_data, valid_data, test_data, batch_size = 32):
    train_loader = data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
    valid_loader = data.DataLoader(valid_data, batch_size = batch_size, shuffle = False)
    test_loader = data.DataLoader(test_data, batch_size = batch_size, shuffle = False)
    
    return train_loader, valid_loader, test_loader