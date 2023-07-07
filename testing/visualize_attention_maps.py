# Project:
#   Localized Questions in VQA
# Description:
#  Visualization of attention maps
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern

import sys
sys.path.append('/home/sergio814/Documents/PhD/code/locvqa/')

import comet_ml 
import torch
from misc import io
from os.path import join as jp
from metrics import metrics
from tqdm import tqdm
from plot import visualize_att
from misc import printer, dirs
from core.datasets import loaders_factory
from core.models import model_factory
from core.train_vault import criteria, optimizers, train_utils, looper

args = io.get_config_file_name()

def main():
    # read config file
    config = io.read_config(args.path_config)

    config['train_from'] = 'best' # set this parameter to best so that best model is loaded for validation part
    config['comet_ml'] = False
    model_name = config['model']

    device = torch.device('cuda' if torch.cuda.is_available() and config['cuda'] else 'cpu')

    # get loaders
    _, vocab_words, vocab_answers, index_unk_answer = loaders_factory.get_vqa_loader('train', config, shuffle=True) 
    test_loader = loaders_factory.get_vqa_loader('test', config, shuffle=False)

    # get model
    model = model_factory.get_vqa_model(config, vocab_words, vocab_answers)

    # create optimizer
    optimizer, scheduler = optimizers.get_optimizer(config, model, add_scheduler=True)

    # get best epoch
    best_epoch, _, _, _, path_logs = train_utils.initialize_experiment(config, model, optimizer, args.path_config, lower_is_better=True)

    dirs.create_folder(jp(path_logs, 'att_maps'))

    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(test_loader)):
            print('Batch', i+1, '/', len(test_loader))
            # move data to GPU
            question = sample['question'].to(device)
            visual = sample['visual'].to(device)
            answer = sample['answer'].to(device)
            question_indexes = sample['question_id'] # keep in cpu
            mask = sample['mask'].to(device)     

            visualize_att.plot_attention_maps(model_name, model, visual, question, mask, answer, vocab_words, path_logs, question_indexes, vocab_answers)       

if __name__ == '__main__':
    main()
