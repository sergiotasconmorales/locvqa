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
import random
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as jp
from PIL import Image
from collections import Counter
from plot import plotter


path_base = '/home/sergio814/Documents/PhD/code/data/Tools/STS2017_v1'
path_output = jp(path_base, 'test_output')
os.makedirs(path_output, exist_ok=True)
path_qa = jp(path_base, 'qa')
path_images = jp(path_base, 'images')

subset = 'train'
n_examples = 100

# load questions
path_questions = jp(path_qa, subset + '_qa.json')
with open(path_questions, 'r') as f:
    qa = json.load(f)

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
    path_image = jp(path_images, subset, example['image_name'])
    image = np.array(Image.open(path_image))
    # build mask from coordinates
    mask = np.zeros(example['mask_size'], dtype=np.uint8)
    mask[example['mask_coords'][0][0]:example['mask_coords'][0][0] + example['mask_coords'][1], example['mask_coords'][0][1]:example['mask_coords'][0][1] + example['mask_coords'][2]] = 255
    # overlay mask on image, and question and answer as title
    fig, ax = plt.subplots()
    plt.title(example['question'] + ' ' + example['answer'])
    plotter.overlay_mask(image, mask, mask, alpha = 0.3, save = True, path_without_ext=jp(path_output, str(i).zfill(3)), ax=ax, fig = fig)

