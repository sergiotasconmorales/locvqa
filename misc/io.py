# Project:
#   Localized Questions in VQA
# Description:
#   I/O file
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern

import yaml
import argparse 
import pickle
import json
import torch
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from os.path import join as jp

def get_config_file_name(pre_extract = False, single=False):
    """Function to create CLI argument parser and return corresponding args

    pre_extract 
    
    single
    Whether or not a single image is to be processed.

    Returns
    -------
    parser
        argument parser
    """
    parser = argparse.ArgumentParser(
        description='Read config file name',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--path_config', default='vqa/config/vqa2/default.yaml', type=str, help='path to a yaml options file') 

    if pre_extract:
        parser.add_argument('--subset', default='train', type=str, help='subset to be processed') 

    if single:
        parser.add_argument('--path_image', default='/home/sergio814/Documents/PhD/code/data/to_inpaint/grade_0/inpainted/IDRiD_133.jpg', type=str, help='Path to image')
        parser.add_argument('--path_mask', default='/home/sergio814/Documents/PhD/code/data/dme_dataset_8_balanced/masks/train/maskA/whole_image_mask.tif', type=str, help='Path to mask')
        parser.add_argument('--question', default='What is the diabetic macular edema grade for this image?', type=str)
        parser.add_argument('--path_output', default=os.getcwd())
    return parser.parse_args()


def read_image(path):
    # function to read an image using PIL Image
    return np.array(Image.open(path))

def save_image(image, path):
    # function to save an image using PIL Image
    Image.fromarray(image).save(path)

def read_weights(config):
    # Function to read (class) weights that come from the answer distribution and were previously computed using compute_answer_weights.py
    path_weights = jp(config['path_data'], 'answer_weights', 'w.pt')
    if not os.path.exists(path_weights):
        raise FileNotFoundError
    weights = torch.load(path_weights)
    return weights

def read_config(path_config):
    """Function to read the config file from path_config

    Parameters
    ----------
    path_config : str
        path to config file

    Returns
    -------
    dict
        parsed config file
    """
    with open(path_config, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg


def save_pickle(data, path):
    """Function to save a pickle file in the specified path

    Parameters
    ----------
    data : list
        data to be saved 
    path : str
        path including format for pickle file
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def read_pickle(path):
    """Function to read a pickle file from the specified path

    Parameters
    ----------
    path : str
        path including format for pickle file

    Returns
    -------
    list
        data read from pickle file
    """
    with open(path, 'rb') as f:
        return pickle.load(f)