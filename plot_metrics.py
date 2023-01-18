# Project:
#   Localized Questions in VQA
# Description:
#   Script for metrics plotting
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern

from os.path import join as jp
import misc.io as io
import torch
import json
import pickle
import os
from plot import plotter
from metrics import metrics 
from misc import general 
import matplotlib.pyplot as plt

torch.manual_seed(1234) # use same seed for reproducibility

# read config name from CLI argument --path_config
args = io.get_config_file_name()

def main():
    # read config file
    config = io.read_config(args.path_config)

    config_file_name = args.path_config.split("/")[-1].split(".")[0]

    path_logs = jp(config['logs_dir'], config['dataset'], config_file_name)

    # first, plot logged learning curves for all available metrics
    with open(jp(path_logs, 'logbook.json'), 'r') as f:
        logbook = json.load(f)

    general_info = logbook['general']
    train_metrics = logbook['train']
    val_metrics = logbook['val']

    #* assumption: all reported train metrics were also reported for validation

    for (k_train, v_train), (k_val, v_val) in zip(train_metrics.items(), val_metrics.items()):
        assert k_train.split('_')[0] == k_val.split('_')[0] # check that metrics correspond
        metric_name = k_train.split('_')[0]

        plotter.plot_learning_curve(v_train, v_val, metric_name, title=general_info['config']['model'] + ' ' + config_file_name, save=True, path=path_logs)

    # if model is binary, plot ROC and PRC curves along with AUC and AP
    if config['num_answers'] == 2:
        # VAL
        # now go to answers folder and read info from there
        path_val_answers_file = jp(path_logs, 'answers', 'answers_epoch_val.pt')
        answers_best_val_epoch = torch.load(path_val_answers_file, map_location=torch.device('cpu')) # dictionary with keys: results, answers. results contains tensor with (question_index, model's answer), answers is  (target, prob)

        auc_val, ap_val, roc_val, prc_val = metrics.compute_roc_prc(answers_best_val_epoch['answers'])
        plotter.plot_roc_prc(roc_val, auc_val, prc_val, ap_val, title='Validation plots', save=True, path=path_logs, suffix='val')

        # TEST
        # plot curves for test set, if it has been processed with inference.py
        path_test_answers_file = jp(path_logs, 'answers', 'answers_epoch_test.pt')
        if not os.path.exists(path_test_answers_file):
            raise Exception("Test set answers haven't been generated with inference.py")
        answers_test = torch.load(path_test_answers_file, map_location=torch.device('cpu'))

        auc_test, ap_test, roc_test, prc_test = metrics.compute_roc_prc(answers_test['answers'])
        plotter.plot_roc_prc(roc_test, auc_test, prc_test, ap_test, title='Test plots', save=True, path=path_logs, suffix='test')


if __name__ == '__main__':
    main()