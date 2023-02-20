# Project:
#   Localized Questions in VQA
# Description:
#   Performance assessment
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern


import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

def vqa_accuracy(predicted, true):
    """ Compute the accuracies for a batch according to VQA challenge accuracy"""
    # in this case true is a [B, 10] matrix where ever row contains all answers for the particular question
    _, predicted_index = predicted.max(dim=1, keepdim=True) # should be [B, 1] where every row is an index
    agreement = torch.eq(predicted_index.view(true.size(0),1), true).sum(dim=1) # row-wise check of times that answer in predicted_index is in true

    return torch.min(agreement*0.3, torch.ones_like(agreement)).float().sum() # returning batch sum

def accuracy(pred, gt):
    return torch.eq(pred, gt).sum()/pred.shape[0]

def batch_strict_accuracy(predicted, true):
    # in this case true is a [B] tensor with the answers 
    sm = nn.Softmax(dim=1)
    probs = sm(predicted)
    _, predicted_index = probs.max(dim=1) # should be [B, 1] where every row is an index
    return torch.eq(predicted_index, true).sum() # returning sum

def batch_binary_accuracy(predicted, true):
    # input predicted already contains the indexes of the answers
    return torch.eq(predicted, true).sum() # returning sum

def compute_auc_ap(targets_and_preds):
    # input is an Nx2 tensor where the first column contains the target answer for all samples and the second column containes the sigmoided predictions
    targets_and_preds_np = targets_and_preds.cpu().numpy()
    auc = roc_auc_score(targets_and_preds_np[:,0], targets_and_preds_np[:,1]) # eventually take np.ones((targets_and_preds_np.shape[0],)) - targets_and_preds_np[:,1]
    ap = average_precision_score(targets_and_preds_np[:,0], targets_and_preds_np[:,1], pos_label=1)
    return auc, ap

def compute_roc_prc(targets_and_preds, positive_label = 1):
    y_true = targets_and_preds[:,0]
    y_pred = targets_and_preds[:,1]
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_pred, pos_label=positive_label)
    ap = average_precision_score(y_true, y_pred, pos_label=positive_label)
    return auc, ap, (fpr, tpr, thresholds_roc), (precision, recall, thresholds_pr)