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
from collections import Counter

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
    else: # for dme, compute accuracies for each type of question
        path_val_answers_file = jp(path_logs, 'answers', 'answers_epoch_val.pt')
        answers_best_val_epoch = torch.load(path_val_answers_file, map_location=torch.device('cpu')) # contains two columns, first one is question id and second one is the predicted answer
        id2pred = {answers_best_val_epoch[i,0].item(): answers_best_val_epoch[i,1].item() for i in range(answers_best_val_epoch.shape[0])}
        # open qa file to get question types
        path_qa_file_val = jp(config['path_data'], 'processed', 'valset.pickle')
        with open(path_qa_file_val, 'rb') as f:
            qa_val = pickle.load(f)
        # add prediction to qa_val
        for q in qa_val:
            q['prediction'] = id2pred[q['question_id']]
        # group questions by type
        types_counts = Counter([e['question_type'] for e in qa_val]).most_common()
        question_types = {e[0]:e[1] for e in types_counts}
        indexes_types = {e[0]:0 for e in types_counts}
        groups_type = {k:torch.zeros((v,2)) for k,v in question_types.items()}
        all_types = torch.zeros((len(qa_val),2))
        # fill groups_type with targets and predictions
        for i, q in enumerate(qa_val):
            groups_type[q['question_type']][indexes_types[q['question_type']],0] = q['answer_index']
            groups_type[q['question_type']][indexes_types[q['question_type']],1] = q['prediction']
            indexes_types[q['question_type']] += 1
            all_types[i,0] = q['answer_index']
            all_types[i,1] = q['prediction']
        # compute accuracy for each type
        accuracies = {k:torch.eq(v[:,0], v[:,1]).sum()/v.shape[0] for k,v in groups_type.items()}
        # print accuracies
        print(config_file_name)
        print('Validation accuracies by type:')
        for k,v in accuracies.items():
            print(f'{k}: {100*v:.2f}')
        print('Overall accuracy: {:.2f}'.format(100*torch.eq(all_types[:,0], all_types[:,1]).sum()/all_types.shape[0]))
        print('*'*50)

        # do exactly the same for test set
        path_test_answers_file = jp(path_logs, 'answers', 'answers_epoch_test.pt')
        if not os.path.exists(path_test_answers_file):
            raise Exception("Test set answers haven't been generated with inference.py")
        answers_test = torch.load(path_test_answers_file, map_location=torch.device('cpu'))
        id2pred = {answers_test[i,0].item(): answers_test[i,1].item() for i in range(answers_test.shape[0])}
        # open qa file to get question types
        path_qa_file_test = jp(config['path_data'], 'processed', 'testset.pickle')
        with open(path_qa_file_test, 'rb') as f:
            qa_test = pickle.load(f)
        # add prediction to qa_val
        for q in qa_test:
            q['prediction'] = id2pred[q['question_id']]
        # group questions by type
        types_counts = Counter([e['question_type'] for e in qa_test]).most_common()
        question_types = {e[0]:e[1] for e in types_counts}
        indexes_types = {e[0]:0 for e in types_counts}
        groups_type = {k:torch.zeros((v,2)) for k,v in question_types.items()}
        all_types = torch.zeros((len(qa_test),2))
        # fill groups_type with targets and predictions
        for i, q in enumerate(qa_test):
            groups_type[q['question_type']][indexes_types[q['question_type']],0] = q['answer_index']
            groups_type[q['question_type']][indexes_types[q['question_type']],1] = q['prediction']
            indexes_types[q['question_type']] += 1
            all_types[i,0] = q['answer_index']
            all_types[i,1] = q['prediction']
        # compute accuracy for each type
        accuracies = {k:torch.eq(v[:,0], v[:,1]).sum()/v.shape[0] for k,v in groups_type.items()}
        # print accuracies
        print('Test accuracies by type:')
        for k,v in accuracies.items():
            print(f'{k}: {100*v:.2f}')
        print('Overall accuracy: {:.2f}'.format(100*torch.eq(all_types[:,0], all_types[:,1]).sum()/all_types.shape[0]))
        print('*'*50)


if __name__ == '__main__':
    main()