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
from PIL import Image, ImageDraw
import torchvision.transforms as T
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from copy import deepcopy

from misc import io
from . import nlp
from . import visual as vis


class VQABase(Dataset):
    def __init__(self, subset, config, dataset_visual):
        self.subset = subset 
        self.config = config
        self.dataset_visual = dataset_visual
        self.path_annotations_and_questions = jp(config['path_data'], 'qa')
        self.path_processed = jp(config['path_data'], 'processed')
        if 'mask_as_text' in config:
            self.mask_as_text = config['mask_as_text']
        else:
            self.mask_as_text = False
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
    def __init__(self, subset, config, dataset_visual, draw_regions=False):
        super().__init__(subset, config, dataset_visual)
        self.augment = config['augment']
        self.draw_regions = draw_regions

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
        # mask_coords has the format ((y,x), h, w)
        if self.config['dataset'] == 'dme': # requires ellipse regions
            mask_ref = Image.new('L', mask_size, 0)
            mask = ImageDraw.Draw(mask_ref)
            mask.ellipse([(mask_coords[0][1], mask_coords[0][0]),(mask_coords[0][1] + mask_coords[2], mask_coords[0][0] + mask_coords[1])], fill=1)
            mask = torch.from_numpy(np.array(mask_ref))
        else:
            mask = torch.zeros(mask_size, dtype=torch.uint8)
            mask[mask_coords[0][0]:mask_coords[0][0]+mask_coords[1] , mask_coords[0][1]:mask_coords[0][1]+mask_coords[2]] = 1
        return mask.unsqueeze_(0)

    def draw_region(self, img, coords, r=2):
        if self.config['dataset'] == 'dme': # requires ellipse regions
            img_ref = T.ToPILImage()(img)
            ((y,x), h, w) = coords
            draw = ImageDraw.Draw(img_ref)
            draw.ellipse([(x, y),(x + w, y + h)], outline='red')
            img_ref = np.array(img_ref)
            img_ref = img_ref.transpose(2,0,1)
            img_ref = torch.from_numpy(img_ref)
            return img_ref
        else:
            ((y,x), h, w) = coords

            for i in range(3):
                img[i, y-r:y+h+r, x-r:x+r] = 0
                img[i, y-r:y+r, x-r:x+w+r] = 0
                img[i, y-r:y+h+r, x+w-r:x+w+r] = 0
                img[i, y+h-r:y+h+r, x-r:x+w+r] = 0

            # set red channel line to red
            img[0, y-r:y+h+r, x-r:x+r] = 1
            img[0, y-r:y+r, x-r:x+w+r] = 1
            img[0, y-r:y+h+r, x+w-r:x+w+r] = 1
            img[0, y+h-r:y+h+r, x-r:x+w+r] = 1
            return img

    def get_by_question_id(self, question_id):
        for i in range(len(self.dataset_qa)):
            if self.dataset_qa[i]['question_id'] == question_id:
                return self.__getitem__(i)

    def regenerate(self, question_ids):
        # reduce self.dataset_qa to only contain question_ids
        temp = [item for item in self.dataset_qa if item['question_id'] in question_ids]
        self.dataset_qa = temp

    # override getitem method
    def __getitem__(self, index):
        sample = {}

        # get qa pair
        item_qa = self.dataset_qa[index]

        # get visual
        visual = self.dataset_visual.get_by_name(item_qa['image_name'])['visual']
        if self.draw_regions:
            # first, apply inverse transform to get original image
            visual = vis.default_inverse_transform()(visual)
            visual = self.draw_region(visual, item_qa['mask_coords'])
            visual = vis.default_transform(self.config['size'])(T.ToPILImage()(visual))
        mask = self.get_mask(item_qa['mask_coords'], item_qa['mask_size'])

        if self.augment:
            sample['visual'], sample['mask'] = self.transform(visual, mask, 448)
        else:
            sample['visual'] = visual
            sample['mask'] = mask

        # get question
        sample['question_id'] = item_qa['question_id']

        # if mask should be included in the questions

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

        if self.mask_as_text:   
            # exchange question_alt to question
            for data in tqdm([data_train, data_val, data_test], desc='mask_as_text is set to True, therefore alt questions are used'):
                for item in data:
                    item['question'], item['question_alt'] = item['question_alt'], item['question']

        sets, maps = nlp.process_qa(self.config, data_train, data_val, data_test, alt_questions=self.mask_as_text)

        # define paths to save pickle files
        if not os.path.exists(self.path_processed):
            os.mkdir(self.path_processed)
        for name, data in sets.items():
            io.save_pickle(data, jp(self.path_processed, name + '.pickle'))
        for name, data in maps.items():
            io.save_pickle(data, jp(self.path_processed, name + '.pickle'))



def get_vqa_dataset(subset, config, dataset_visual, draw_regions=False):
    # provides dataset class for current training config
    if config['dataset'] in ['cholec', 'sts2017', 'insegcat', 'dme']:
        dataset_vqa = VQARegionsSingle(subset, config, dataset_visual, draw_regions=draw_regions)
    elif config['dataset'] == 'IDRID':
        raise NotImplementedError
    else:
        dataset_vqa = VQABase(subset, config, dataset_visual)

    return dataset_vqa