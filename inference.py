# Project:
#   Localized Questions in VQA
# Description:
#   Inference script to get results and plots
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern

import comet_ml 
import torch
import misc.io as io
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

if __name__ == '__main__':
    main()