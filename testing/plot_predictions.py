# Project:
#   Localized Questions in VQA
# Description:
#  Script to plot some predictions from a model
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern

import sys
sys.path.append('/home/sergio814/Documents/PhD/code/locvqa/')

import os
import json
import pickle
import random
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as jp
from PIL import Image
from collections import Counter
from plot import plotter

DATASET_NAME = {'cholec': 'CholecVQA', 'sts2017': 'STS2017'}

dataset = 'cholec'
version = 'v1'
subset = 'test' # 'val' or 'test'
config = '003'
num_examples = 50 # number of examples to plot

# define paths
path_data = '/home/sergio814/Documents/PhD/code/data/Tools/' + DATASET_NAME[dataset] + '_' + version
path_qa = jp(path_data, 'processed')
path_images = jp(path_data, 'images')
path_logs = '/home/sergio814/Documents/PhD/code/logs/' + dataset + '/config_' + config
path_output = jp(path_logs, 'prediction_examples')
os.makedirs(path_output, exist_ok=True)
os.makedirs(jp(path_output, subset), exist_ok=True)

# load questions
with open(jp(path_qa, subset + 'set.pickle'), 'rb') as f:
    qa = pickle.load(f)

# load dictionary idx to answer
with open(jp(path_qa, 'map_index_answer.pickle'), 'rb') as f:
    idx2answer = pickle.load(f)

# load predictions
preds = torch.load(jp(path_logs, 'answers', 'answers_epoch_' + subset + '.pt'))['results'] # using results field, which has question_id and answer based on 0.5 threshold
question_id2pred = {preds[i,0].item(): preds[i,1].item() for i in range(preds.shape[0])}

# add predictions to qa
for q in tqdm(qa):
    q['prediction'] = idx2answer[question_id2pred[q['question_id']]]

# for now, I will focus on errouneous predictions
qa_wrong = [q for q in qa if q['prediction'] != q['answer']]
for i in range(num_examples):
    example = random.choice(qa_wrong)
    id_example = example['question_id']
    path_image = jp(path_images, subset, example['image_name'])
    image = np.array(Image.open(path_image))
    mask = np.zeros(example['mask_size'], dtype=np.uint8)
    mask[example['mask_coords'][0][0]:example['mask_coords'][0][0] + example['mask_coords'][1], example['mask_coords'][0][1]:example['mask_coords'][0][1] + example['mask_coords'][2]] = 255
    fig, ax = plt.subplots()
    plt.title(example['question'] + '\n'
             + 'GT: ' + example['answer'] + '\n'
             + 'Pred: ' + example['prediction'] + '\n'
             + 'Question id: ' + str(example['question_id']))
    plotter.overlay_mask(image, mask, mask, alpha = 0.3, save = True, path_without_ext=jp(path_output, subset, str(i).zfill(3)), ax=ax, fig = fig)

