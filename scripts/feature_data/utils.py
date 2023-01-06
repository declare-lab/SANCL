import os
import shutil

import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision import datasets, models, transforms
from tqdm import tqdm


class PathImageFolder(datasets.ImageFolder):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return path, sample


def allocate_fn(samples):
    batch = len(samples)
    path = [samples[i][0] for i in range(batch)]
    sample = torch.stack([samples[i][1] for i in range(batch)], dim=0)
    return path, sample


def get_feature_dataloader(pic_root, transform, batch_size):
    img_dataset = PathImageFolder(root=pic_root, transform=transform)
    img_loader = torch.utils.data.DataLoader(img_dataset,
                                             collate_fn=allocate_fn,
                                             batch_size=batch_size)
    return img_loader


def load_img(img_path):
    input_image = Image.open(img_path)
    return input_image
