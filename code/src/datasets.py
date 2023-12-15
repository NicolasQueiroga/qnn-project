import os
import shutil
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.images = self.load_images()
        self.transform = transform

    def load_images(self):
        images = []
        for c in self.classes:
            class_path = os.path.join(self.root_dir, c)
            class_idx = self.class_to_idx[c]
            for filename in os.listdir(class_path):
                if filename.endswith(".jpg"):
                    image_path = os.path.join(class_path, filename)
                    images.append((image_path, class_idx))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


def split_images(
    images_dir, train_dir, test_dir, val_dir, train_ratio=0.8, val_ratio=0.1
):
    print("Splitting images...")
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)

    os.makedirs(train_dir)
    os.makedirs(test_dir)
    os.makedirs(val_dir)

    for c in os.listdir(images_dir):
        class_dir = os.path.join(images_dir, c)
        images = os.listdir(class_dir)

        n_train = int(len(images) * train_ratio)
        n_val = int(n_train * val_ratio)
        n_train = int(n_train - n_val)

        train_images = images[:n_train]
        val_images = images[n_train : n_train + n_val]
        test_images = images[n_train + n_val :]

        os.makedirs(os.path.join(train_dir, c), exist_ok=True)
        os.makedirs(os.path.join(test_dir, c), exist_ok=True)
        os.makedirs(os.path.join(val_dir, c), exist_ok=True)

        for image in train_images:
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(train_dir, c, image)
            shutil.copyfile(image_src, image_dst)

        for image in val_images:
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(val_dir, c, image)
            shutil.copyfile(image_src, image_dst)

        for image in test_images:
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(test_dir, c, image)
            shutil.copyfile(image_src, image_dst)

    print("Images splitted.\n")


def get_train_transforms(img_size, means, stds):
    print("Getting train transforms...")
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomCrop(img_size, padding=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=stds),
        ]
    )


def get_test_transforms(img_size, means, stds):
    print("Getting test transforms...")
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=stds),
        ]
    )


def pretrained_info(train_dir):
    print("Getting pretrained means and stds...")
    train_data = datasets.ImageFolder(root=train_dir, transform=transforms.ToTensor())

    means = torch.zeros(3)
    stds = torch.zeros(3)

    for img, _ in train_data:
        means += torch.mean(img, dim=(1, 2))
        stds += torch.std(img, dim=(1, 2))

    means /= len(train_data)
    stds /= len(train_data)

    return np.array(means), np.array(stds)


def get_datasets(
    image_dir,
    train_dir,
    test_dir,
    val_dir,
    img_size=256,
    train_ratio=0.8,
    val_ratio=0.1,
):
    split_images(image_dir, train_dir, test_dir, val_dir, train_ratio, val_ratio)

    means, stds = pretrained_info(train_dir)

    train_transform = get_train_transforms(img_size, means, stds)
    test_transform = get_test_transforms(img_size, means, stds)

    train_dataset = CustomDataset(train_dir, transform=train_transform)
    test_dataset = CustomDataset(test_dir, transform=test_transform)
    val_dataset = CustomDataset(val_dir, transform=test_transform)

    return train_dataset, test_dataset, val_dataset


def get_dataloaders(
    train_dataset,
    test_dataset,
    val_dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
):
    print("Getting dataloaders...")
    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_dataloader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dataloader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_dataloader, test_dataloader, val_dataloader
