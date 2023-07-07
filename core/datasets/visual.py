# Project:
#   Localized Questions in VQA
# Description:
#   Visual dataset handling
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern

import os
import h5py
import torch
from os.path import join as jp
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class ImagesDataset(Dataset):

    def __init__(self, subset, config, transform=None):
        self.subset = subset
        self.path_img = jp(config['path_data'], 'images', subset)
        self.transform = transform
        self.images = os.listdir(self.path_img) # list all images in folder
        self.map_name_index = {img:i for i, img in enumerate(self.images)}
        self.map_index_name = self.images

    def get_by_name(self, image_name):
        return self.__getitem__(self.map_name_index[image_name])

    def __getitem__(self, index):
        sample = {} 
        sample['name'] = self.map_index_name[index]
        sample['path'] = jp(self.path_img, sample['name']) # full relative path to image
        sample['visual'] = Image.open(sample['path']).convert('RGB')

        # apply transform(s)
        if self.transform is not None:
            sample['visual'] = self.transform(sample['visual'])

        return sample

    def __len__(self):
        return len(self.images)


def default_transform(size):
    """Define basic (standard) transform for input images, as required by image processor

    Parameters
    ----------
    size : int or tuple
        new size for the images

    Returns
    -------
    torchvision transform
        composed transform for files
    """
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    return transform

def default_inverse_transform():
    # undoes basic ImageNet normalization
    transform = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                        std = [ 1., 1., 1. ]),
                                ])
    return transform

def get_visual_dataset(subset, config, transform=None):
    """Get visual dataset either from images or from extracted features

    Parameters
    ----------
    split : str
        split name (train, val, test, trainval)
    options_visual : dict
        visual options as determined in yaml file
    transform : torchvision transform, optional
        transform to be applied to images, by default None

    Returns
    -------
    images dataset
        images dataset with images or feature maps (depending on options_visual['mode'])
    """

    if transform is None:
        transform = default_transform(config['size'])
    visual_dataset = ImagesDataset(subset, config, transform) # create images dataset
    return visual_dataset