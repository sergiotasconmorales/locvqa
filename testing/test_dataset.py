# Project:
#   Localized Questions in VQA
# Description:
#   Script to test a dataset in terms of question_id unicity, balance of answers, and visualization of random examples against the GT.
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern

import sys
sys.path.append('/home/sergio814/Documents/PhD/code/locvqa/')

import os
import json
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as jp
from PIL import Image
from collections import Counter
from plot import plotter


path_base = '/home/sergio814/Documents/PhD/code/data/Tools/INSEGCAT_v1'
path_output = jp(path_base, 'test_output')
os.makedirs(path_output, exist_ok=True)
path_qa = jp(path_base, 'qa')
path_processed = jp(path_base, 'processed')
path_images = jp(path_base, 'images')

subset = 'test'
os.makedirs(jp(path_output, subset), exist_ok=True)
n_examples = 50

# load questions
path_questions = jp(path_qa, subset + '_qa.json')
with open(path_questions, 'r') as f:
    qa = json.load(f)

path_processed_qa = jp(path_processed, subset + 'set.pickle')
with open(path_processed_qa, 'rb') as f:
    processed_qa = pickle.load(f)

# build dict question_id to entry from processed_qa
processed_qa_dict = {q['question_id']: q for q in processed_qa}

# first, check unicity of question ids
question_ids = [q['question_id'] for q in qa]
if len(question_ids) == len(set(question_ids)):
    print('PASSED: All question ids are unique')
else:
    print('FAILED: There are repeated question ids')

# second, check balance of answers
answers = [q['answer'] for q in qa]
answer_counts = Counter(answers).most_common()
print('Answer counts:')
for answer, count in answer_counts:
    print(answer, count)

# third, visualize random examples
print('Generating random examples. To be saved at', path_output)
for i in range(n_examples):
    example = random.choice(qa)
    id_example = example['question_id']
    processed_example = processed_qa_dict[id_example]
    path_image = jp(path_images, subset, example['image_name'])
    image = np.array(Image.open(path_image))
    # build mask from coordinates
    mask = np.zeros(example['mask_size'], dtype=np.uint8)
    mask[example['mask_coords'][0][0]:example['mask_coords'][0][0] + example['mask_coords'][1], example['mask_coords'][0][1]:example['mask_coords'][0][1] + example['mask_coords'][2]] = 255
    # overlay mask on image, and question and answer as title
    fig, ax = plt.subplots()
    plt.title(example['question'] + ' ' + example['answer'])
    # compare information from qa and processed_qa
    print('Question qa:', example['question'], 'Question processed_qa:', processed_example['question'])
    print('Image name qa:', example['image_name'], 'Image name processed_qa:', processed_example['image_name'])
    print('Answer qa:', example['answer'], 'Answer processed_qa:', processed_example['answer'])
    print('Mask coords qa:', example['mask_coords'], 'Mask coords processed_qa:', processed_example['mask_coords'])
    print('Mask size qa:', example['mask_size'], 'Mask size processed_qa:', processed_example['mask_size'])
    plotter.overlay_mask(image, mask, mask, alpha = 0.3, save = True, path_without_ext=jp(path_output, subset, str(i).zfill(3)), ax=ax, fig = fig)

