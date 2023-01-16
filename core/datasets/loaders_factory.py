# Project:
#   Localized Questions in VQA
# Description:
#   Script to provide dataloaders
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern

import numpy as np
from torch.utils.data import DataLoader
import collections
import torch
from . import visual, vqa, aux


def get_vqa_loader(subset, config, shuffle=False):

    # create visual dataset for images
    dataset_visual = visual.get_visual_dataset(subset, config)

    # create vqa dataset for questions and answers
    dataset_vqa = vqa.get_vqa_dataset(subset, config, dataset_visual)

    dataloader = DataLoader(    dataset_vqa,
                                batch_size = config['batch_size'],
                                shuffle=shuffle,
                                num_workers=config['num_workers'],
                                pin_memory=config['pin_memory'],
                                collate_fn=aux.collater
                            )
    if subset == 'train':
        return dataloader, dataset_vqa.map_index_word, dataset_vqa.map_index_answer, dataset_vqa.index_unknown_answer
    else:
        return dataloader