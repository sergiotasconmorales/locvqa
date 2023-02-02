# Project:
#   Localized Questions in VQA
# Description:
#   Optimizers
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_optimizer(config, model, add_scheduler=False):

    if 'adam' in config['optimizer']:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), config['learning_rate'])
    elif 'adadelta' in config['optimizer']:
        optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), config['learning_rate'])
    elif 'rmsprop' in config['optimizer']:
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), config['learning_rate'])
    elif 'sgd' in config['optimizer']:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), config['learning_rate'])

    if add_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, 'min')
        return optimizer, scheduler
    else:
        return optimizer