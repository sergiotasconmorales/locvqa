# Project:
#   Localized Questions in VQA
# Description:
#  Script to plot metrics per class
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern

import sys
sys.path.append('/home/sergio814/Documents/PhD/code/locvqa/')

from os.path import join as jp
import misc.io as io
import torch
import json
from tqdm import tqdm
import os
from plot import plotter
from metrics import metrics 
import numpy as np

# read config name from CLI argument --path_config
args = io.get_config_file_name()

def main():
    # read config file
    config = io.read_config(args.path_config)

    config_file_name = args.path_config.split("/")[-1].split(".")[0]

    path_logs = jp(config['logs_dir'], config['dataset'], config_file_name)

    path_qa = jp(config['path_data'], 'processed')

    # read qa test using pickle
    qa_test = io.read_pickle(jp(path_qa, 'testset.pickle'))
    # read qa val using pickle
    qa_val = io.read_pickle(jp(path_qa, 'valset.pickle'))
    # read map answer to index
    map_answer2idx = io.read_pickle(jp(path_qa, 'map_answer_index.pickle'))

    if config['num_answers'] == 2:
        path_test_answers_file = jp(path_logs, 'answers', 'answers_epoch_test.pt')
        if not os.path.exists(path_test_answers_file):
            raise Exception("Test set answers haven't been generated with inference.py")
        answers_test = torch.load(path_test_answers_file, map_location=torch.device('cpu'))
        # build dictionary with key: answer, value: probability
        id2prob = {answers_test['results'][i,0].item(): answers_test['answers'][i,1].item() for i in range(answers_test['answers'].shape[0])}
        # add probability to qa_test
        for q in qa_test:
            q['prob'] = id2prob[q['question_id']]
        # separate qa_test into groups based on object_object field, putting as value the gt answer and the probability
        objects_classes = set([q['question_object'] for q in qa_test])
        qa_test_per_class = {obj_class: [] for obj_class in objects_classes}
        for q in tqdm(qa_test):
            qa_test_per_class[q['question_object']].append((q['answer_index'], q['prob']))
        # compute metrics per class
        for k, v in qa_test_per_class.items():
            matrix = np.array(v, dtype=float)
            # compute metrics
            auc_test, ap_test, roc_test, prc_test = metrics.compute_roc_prc(matrix)
            plotter.plot_roc_prc(roc_test, auc_test, prc_test, ap_test, title=k, save=True, path=path_logs, suffix=k)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()