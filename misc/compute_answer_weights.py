# Project:
#   Localized Questions in VQA
# Description:
#  Script to compute the weights for the answers
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern

from os.path import join as jp
import os
import pickle
from collections import Counter
import torch

path_base = '/home/sergio814/Documents/PhD/code/data/Tools/DME_v2/' 
path_processed = jp(path_base, 'processed')
path_output = jp(path_base, 'answer_weights')
os.makedirs(path_output, exist_ok=True)
path_output_file = jp(path_output, 'w.pt')

# read train QA pairs using pickle
path_trainset = jp(path_processed, 'trainset.pickle')
with open(path_trainset, 'rb') as f:
    trainset = pickle.load(f)

answers = [e['answer_index'] for e in trainset]

countings = Counter(answers).most_common()
countings_dict = {e[0]:e[1] for e in countings}
weights = torch.zeros(len(countings_dict))
for i in range(weights.shape[0]):
    weights[i] = countings_dict[i]

# normalize weights as suggested in https://discuss.pytorch.org/t/weights-in-weighted-loss-nn-crossentropyloss/69514
weights = 1 - weights/weights.sum()

# save weights to target file
torch.save(weights, path_output_file)