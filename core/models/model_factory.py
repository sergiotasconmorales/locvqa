# Project:
#   Localized Questions in VQA
# Description:
#   Model factory
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern

import torch.nn as nn
from . import models


def get_vqa_model(config, vocab_words, vocab_answers):
    # function to provide a vqa model

    model = getattr(models, config['model'])(config, vocab_words, vocab_answers)

    if config['data_parallel'] and config['cuda']:
        model = nn.DataParallel(model).cuda()

    return model