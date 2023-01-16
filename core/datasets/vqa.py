# Project:
#   Localized Questions in VQA
# Description:
#   VQA dataset classes
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern


import pickle
import os 
import json
import random
from os.path import join as jp
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from misc import io
from . import nlp



class VQABase(Dataset):
    def __init__(self, subset, config, dataset_visual):
        self.subset = subset 
        self.config = config
        self.dataset_visual = dataset_visual
        self.path_annotations_and_questions = jp(config['path_data'], 'qa')
        self.path_processed = jp(config['path_data'], 'processed')
        if not os.path.exists(self.path_processed) or len(os.listdir(self.path_processed))<1 or (subset == 'train' and config['process_qa_again']):
            self.pre_process_qa() # pre-process qa, produce pickle files

        # load pre-processed qa
        self.read_prep_rocessed(self.path_processed)

    def pre_process_qa(self):
        raise NotImplementedError # to be implemented in baby class

    def read_prep_rocessed(self, path_files):
        # define paths
        path_map_index_word = jp(path_files, 'map_index_word.pickle')
        path_map_word_index = jp(path_files, 'map_word_index.pickle')
        path_map_index_answer = jp(path_files, 'map_index_answer.pickle')
        path_map_answer_index = jp(path_files, 'map_answer_index.pickle')
        path_dataset = jp(path_files, self.subset + 'set.pickle')

        # read files
        with open(path_map_index_word, 'rb') as f:
                    self.map_index_word = pickle.load(f)
        with open(path_map_word_index, 'rb') as f:
                    self.map_word_index = pickle.load(f)
        with open(path_map_index_answer, 'rb') as f:
                    self.map_index_answer = pickle.load(f)
        with open(path_map_answer_index, 'rb') as f:
                    self.map_answer_index = pickle.load(f)
        with open(path_dataset, 'rb') as f:
                    self.dataset_qa = pickle.load(f)

        # save unknown answer index 
        self.index_unknown_answer = self.map_answer_index['UNK']

    def __getitem__(self, index):
        sample = {}

        # get qa pair
        item_qa = self.dataset_qa[index]

        # get visual
        sample['visual'] = self.dataset_visual.get_by_name(item_qa['image_name'])['visual']

        # get question
        sample['question_id'] = item_qa['question_id']
        sample['question'] = torch.LongTensor(item_qa['question_word_indexes'])

        # get answer
        sample['answer'] = item_qa['answer_index']
        if 'answer_indexes' in item_qa: # trick so that this class can be used with non-vqa2 data
            sample['answers'] = item_qa['answers_indexes']

        return sample

    def __len__(self):
        return len(self.dataset_qa)


class VQARegionsSingle(VQABase):
    """Class for dataloader that contains questions about a single region

    Parameters
    ----------
    VQABase : Parent class
        Base class for VQA dataset.
    """
    def __init__(self, subset, config, dataset_visual):
        super().__init__(subset, config, dataset_visual)
        self.augment = config['augment']
        self.mask_as_text = config['mask_as_text']

    def transform(self, image, mask, size):

        if self.subset == 'train': # only for training samples

            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            ## random rotation in small range
            #if random.random() > 0.5:
            #    angle = random.randint(-10, 10) 
            #    image = TF.rotate(image, angle)
            #    mask = TF.rotate(mask, angle)

        # Transform to tensor
        if not torch.is_tensor(image):
            image = TF.to_tensor(image)
        if not torch.is_tensor(mask):
            mask = TF.to_tensor(mask)
        return image, mask

    def get_mask(self, mask_coords, mask_size):
        mask = torch.zeros(mask_size, dtype=torch.uint8)
        mask[mask_coords[0][0]:mask_coords[0][0]+mask_coords[1] , mask_coords[0][1]:mask_coords[0][1]+mask_coords[2]] = 1
        return mask.unsqueeze_(0)

    # override getitem method
    def __getitem__(self, index):
        sample = {}

        # get qa pair
        item_qa = self.dataset_qa[index]

        # get visual
        visual = self.dataset_visual.get_by_name(item_qa['image_name'])['visual']
        mask = self.get_mask(item_qa['mask_coords'], item_qa['mask_size'])

        if self.augment:
            sample['visual'], sample['mask'] = self.transform(visual, mask, 448)
        else:
            sample['visual'] = visual
            sample['mask'] = mask

        # get question
        sample['question_id'] = item_qa['question_id']

        # if mask should be included in the questions
        if self.mask_as_text:
            sample['question'] = torch.LongTensor(item_qa['question_word_indexes_alt']) # question with location as text
        else:  
            sample['question'] = torch.LongTensor(item_qa['question_word_indexes'])

        # get answer
        sample['answer'] = item_qa['answer_index']

        return sample

    # define preprocessing method for qa pairs
    def pre_process_qa(self):

        # define paths to save pickle files. Have to process all of them at the same time because the train set determines possible answers and vocabularies
        data_train = json.load(open(jp(self.path_annotations_and_questions, 'train_qa.json'), 'r'))
        data_val = json.load(open(jp(self.path_annotations_and_questions, 'val_qa.json'), 'r'))
        data_test = json.load(open(jp(self.path_annotations_and_questions, 'test_qa.json'), 'r'))

        sets, maps = nlp.process_qa(self.config, data_train, data_val, data_test)

        # define paths to save pickle files
        if not os.path.exists(self.path_processed):
            os.mkdir(self.path_processed)
        for name, data in sets.items():
            io.save_pickle(data, jp(self.path_processed, name + '.pickle'))
        for name, data in maps.items():
            io.save_pickle(data, jp(self.path_processed, name + '.pickle'))



def get_vqa_dataset(subset, config, dataset_visual):
    # provides dataset class for current training config
    if config['dataset'] in ['Cholec', 'sts2017']:
        dataset_vqa = VQARegionsSingle(subset, config, dataset_visual)
    elif config['dataset'] == 'IDRID':
        raise NotImplementedError
    else:
        dataset_vqa = VQABase(subset, config, dataset_visual)

    return dataset_vqa