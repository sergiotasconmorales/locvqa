# Project:
#   Localized Questions in VQA
# Description:
#   Inference script to get results and plots
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern

import comet_ml 
import torch
import misc.io as io
from metrics import metrics
from misc import printer
from core.datasets import loaders_factory
from core.models import model_factory
from core.train_vault import criteria, optimizers, train_utils, looper

args = io.get_config_file_name()

def main():
    # read config file
    config = io.read_config(args.path_config)

    config['train_from'] = 'best' # set this parameter to best so that best model is loaded for validation part
    config['comet_ml'] = False

    device = torch.device('cuda' if torch.cuda.is_available() and config['cuda'] else 'cpu')

    # get loaders
    train_loader, vocab_words, vocab_answers, index_unk_answer = loaders_factory.get_vqa_loader('train', config, shuffle=True) 
    val_loader = loaders_factory.get_vqa_loader('val', config, shuffle=False)
    test_loader = loaders_factory.get_vqa_loader('test', config, shuffle=False)
    
    # get model
    model = model_factory.get_vqa_model(config, vocab_words, vocab_answers)

    if 'weighted_loss' in config:
        if config['weighted_loss']:
            answer_weights = io.read_weights(config) # if use of weights is required, read them from folder where they were previously saved using compute_answer_weights scripts
        else:
            answer_weights = None # If false, just set variable to None
    else:
        answer_weights = None

    # create criterion
    criterion = criteria.get_criterion(config, device, ignore_index = index_unk_answer, weights=answer_weights)

    # create optimizer
    optimizer, scheduler = optimizers.get_optimizer(config, model, add_scheduler=True)

    # get best epoch
    best_epoch, _, _, _, path_logs = train_utils.initialize_experiment(config, model, optimizer, args.path_config, lower_is_better=True)

    # get validation function
    _, validate = looper.get_looper_functions(config)

    metrics_test, results_test = validate(test_loader, model, criterion, device, 0, config, None, comet_exp=None)
    print("Test set was evaluated for epoch", best_epoch-1, "which was the epoch with the lowest", config['metric_to_monitor'], "during training")
    print(metrics_test)
    train_utils.save_results(results_test, 'test', config, path_logs) # test results saved as epoch 0

    metrics_val, results_val = validate(val_loader, model, criterion, device, 0, config, None, comet_exp=None)
    print("Metrics after inference on the val set, best epoch")
    print(metrics_val)
    train_utils.save_results(results_val, 'val', config, path_logs)

    # adding code to get results for each type of question
    # get question types
    qa_data_test = test_loader.dataset.dataset_qa
    qa_data_val = val_loader.dataset.dataset_qa
    # i need to generate a dict, where each key is a question type and the value is a tensor where the first row is the answer and the second one is the probability
    # i need to do this for the test and val set
    # first, let's get a dict question_id to question_type
    question_id_to_type_test = {e['question_id']:e['question_type'] for e in qa_data_test}
    types_test = list(set(question_id_to_type_test.values()))
    type2idx_test = {t:i for i,t in enumerate(types_test)}
    question_id_to_type_val = {e['question_id']:e['question_type'] for e in qa_data_val}
    types_val = list(set(question_id_to_type_val.values()))
    type2idx_val = {t:i for i,t in enumerate(types_val)}
    # dicts to store answers to get auc, acc and ap from
    answers_group_test = {}
    answers_group_val = {}   
    typeidx_test = torch.zeros(len(results_test['answers']), dtype=torch.long)
    typeidx_val = torch.zeros(len(results_val['answers']), dtype=torch.long)
    for i in range(typeidx_test.shape[0]):
        typeidx_test[i] = type2idx_test[question_id_to_type_test[results_test['results'][i,0].item()]]
    for i in range(typeidx_val.shape[0]):
        typeidx_val[i] = type2idx_val[question_id_to_type_val[results_val['results'][i,0].item()]]
    # now for each type, get metrics
    for k,v in type2idx_test.items():
        answers_group_test[k] = results_test['answers'][typeidx_test==v]
    for k,v in type2idx_val.items():
        answers_group_val[k] = results_val['answers'][typeidx_val==v]
    # now get metrics
    printer.print_line()
    # test
    for k,v in answers_group_test.items():
        auc_test, ap_test = metrics.compute_auc_ap(v)
        print("AUC for test set, question type", k, "is", auc_test)
        print("AP for test set, question type", k, "is", ap_test)
    # val
    for k,v in answers_group_val.items():
        auc_val, ap_val = metrics.compute_auc_ap(v)
        print("AUC for val set, question type", k, "is", auc_val)
        print("AP for val set, question type", k, "is", ap_val)


if __name__ == '__main__':
    main()