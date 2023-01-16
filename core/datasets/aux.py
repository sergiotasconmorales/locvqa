# Project:
#   Localized Questions in VQA
# Description:
#   Auxiliary functions for dataset and dataloader creation
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern

import torch
import collections
import numpy as np


def collater(batch):
    # function to collate several samples of a batch
    if torch.is_tensor(batch[0]):
        return torch.stack(batch, 0)
    elif type(batch[0]).__module__ == np.__name__ and type(batch[0]).__name__ == 'ndarray':
        return torch.stack([torch.from_numpy(sample) for sample in batch], 0)
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.tensor(batch) # * use DoubleTensor?
    elif isinstance(batch[0], dict):
        res = dict.fromkeys(batch[0].keys())
        for k in res.keys():
            res[k] = [s[k] for s in batch]
        return {k:collater(v) for k,v in res.items()}
    elif isinstance(batch[0], collections.Iterable):
        return torch.tensor(batch, dtype=torch.int) # ! integers because only for all answers it gets here and the indices are integers
    else:
        raise ValueError("Unknown type of samples in the batch. Add condition to collater function")



