# Project:
#   Localized Questions in VQA
# Description:
#   Main train file
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern

# IMPORTANT: All configurations are made through the yaml config file which is located in config/<dataset>/<file>.yaml. The path to this file is
#           specified using CLI arguments, with --path_config <path_to_yaml_file> . If you don't use comet ml, set the parameter comet_ml to False

import time
import comet_ml
import torch 
import misc.io as io
from core.datasets import loaders_factory
from core.models import model_factory
from core.train_vault import criteria, optimizers, train_utils, looper, comet

# read config name from CLI argument --path_config
args = io.get_config_file_name()

def main():
    # read config file
    config = io.read_config(args.path_config)
    if 'draw_regions' in config:
        draw_regions = config['draw_regions']
    else:
        draw_regions = False

    device = torch.device('cuda' if torch.cuda.is_available() and config['cuda'] else 'cpu')

    train_loader, vocab_words, vocab_answers, index_unk_answer = loaders_factory.get_vqa_loader('train', config, shuffle=True, draw_regions=draw_regions) 

    print('Num batches train: ', len(train_loader))
    print('Num samples train:', len(train_loader.dataset))

    val_loader = loaders_factory.get_vqa_loader('val', config, shuffle=False, draw_regions=draw_regions)

    print('Num batches val: ', len(val_loader))
    print('Num samples val:', len(val_loader.dataset))

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

    # initialize experiment
    start_epoch, comet_experiment, early_stopping, logbook, path_logs = train_utils.initialize_experiment(config, model, optimizer, args.path_config, lower_is_better=True)

    # log config info
    logbook.log_general_info('config', config)

    # get train and val functions
    train, validate = looper.get_looper_functions(config)

    # train loop
    for epoch in range(start_epoch, config['epochs']+1):
        print('Epoch: ', epoch)
        # train for one epoch
        train_epoch_metrics = train(train_loader, model, criterion, optimizer, device, epoch, config, logbook, comet_exp=comet_experiment)
        comet.log_metrics(comet_experiment, train_epoch_metrics, epoch)
        # validation
        val_epoch_metrics, val_results = validate(val_loader, model, criterion, device, epoch, config, logbook, comet_exp=comet_experiment)
        comet.log_metrics(comet_experiment, val_epoch_metrics, epoch)
        # run step of scheduler
        scheduler.step(val_epoch_metrics[config['metric_to_monitor']])
        # save validation answers for current epoch
        train_utils.save_results(val_results, epoch, config, path_logs)
        logbook.save_logbook(path_logs)
        # check early stopping condition
        early_stopping(val_epoch_metrics, config['metric_to_monitor'], model, optimizer, epoch)
        # if patience was reached, stop train loop
        if early_stopping.early_stop:
            print("Early stopping")
            break

if __name__ == '__main__':
    main()