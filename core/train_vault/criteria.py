# Project:
#   Localized Questions in VQA
# Description:
#   Loss functions file
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern

from torch import nn
import numpy as np
import torch
import os

def get_criterion(config, device, ignore_index = None, weights = None):
    # function to return a criterion. By default I set reduction to 'sum' so that batch averages are not performed because I want the average across the whole dataset
    if config['loss'] == 'crossentropy':
        if weights is not None:
            crit = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean', weight=weights).to(device)        
        else:
            crit = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='sum').to(device)
    elif config['loss'] == 'bce':
        if weights is not None:
            crit = nn.BCEWithLogitsLoss(reduction='mean').to(device)
        else:
            crit = nn.BCEWithLogitsLoss(reduction='sum').to(device)
    else:
        raise ValueError("Unknown loss function.")

    return crit