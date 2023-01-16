# Project:
#   Localized Questions in VQA
# Description:
#   Train loops
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern

import torch
from torch import nn
from metrics import metrics

def train(train_loader, model, criterion, optimizer, device, epoch, config, logbook, comet_exp=None, consistency_term=None, rels=None):
    # set train mode
    model.train()

    # Initialize variables for collecting metrics from all batches
    loss_epoch = 0.0
    running_loss = 0.0
    acc_epoch = 0.0

    for i, sample in enumerate(train_loader):

        # move data to GPU
        question = sample['question'].to(device)
        visual = sample['visual'].to(device)
        answer = sample['answer'].to(device)
        mask = sample['mask'].to(device)

        # clear parameter gradients
        optimizer.zero_grad()

        # get output from model
        output = model(visual, question, mask)

        # compute loss
        loss = criterion(output, answer)

        loss.backward()

        optimizer.step()


        running_loss += loss.item()
        if comet_exp is not None and i%10 == 9: # log every 10 iterations
            comet_exp.log_metric('loss_train_step', running_loss/10, step=len(train_loader)*(epoch-1) + i+1)
            running_loss = 0.0

        # compute accuracy
        acc = metrics.batch_strict_accuracy(output, answer)

        # laters: save to logger and print 
        loss_epoch += loss.item()
        acc_epoch += acc.item()

    metrics_dict = {'loss_train': loss_epoch/len(train_loader.dataset), 'acc_train': acc_epoch/len(train_loader.dataset)} 

    logbook.log_metrics('train', metrics_dict, epoch)

    return metrics_dict# returning average for all samples


def validate(val_loader, model, criterion, device, epoch, config, logbook, comet_exp=None, consistency_term=None, rels=None):

    #if config['mainsub']:
    #    denominator_acc = 2*len(val_loader.dataset)
    #else:
    denominator_acc = len(val_loader.dataset)

    # tensor to save results
    results = torch.zeros((denominator_acc, 2), dtype=torch.int64)

    # set evaluation mode
    model.eval()

    # Initialize variables for collecting metrics from all batches
    loss_epoch = 0.0
    acc_epoch = 0.0

    offset = 0
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            batch_size = sample['question'].size(0)

            # move data to GPU
            question = sample['question'].to(device)
            visual = sample['visual'].to(device)
            answer = sample['answer'].to(device)
            question_indexes = sample['question_id'] # keep in cpu
            mask = sample['mask'].to(device)
            output = model(visual, question, mask)


            loss = criterion(output, answer)

            # compute accuracy
            acc = metrics.batch_strict_accuracy(output, answer)

            # save answer indexes and answers
            sm = nn.Softmax(dim=1)
            probs = sm(output)
            _, pred = probs.max(dim=1)
        
            results[offset:offset+batch_size,0] = question_indexes
            results[offset:offset+batch_size,1] = pred
            offset += batch_size

            loss_epoch += loss.item()
            acc_epoch += acc.item()

    metrics_dict = {'loss_val': loss_epoch/denominator_acc, 'acc_val': acc_epoch/denominator_acc} 
    if logbook is not None:
        logbook.log_metrics('val', metrics_dict, epoch)

    return metrics_dict, results # returning averages for all samples



# ------------------------------------------------------------------------------------------------
# functions for binary case
# ------------------------------------------------------------------------------------------------


def train_binary(train_loader, model, criterion, optimizer, device, epoch, config, logbook, comet_exp=None, consistency_term=None):
    # tensor to save results
    results = torch.zeros((len(train_loader.dataset), 2), dtype=torch.int64) # to store question id, model's answer
    answers = torch.zeros(len(train_loader.dataset), 2) # store target answer, prob

    # set train mode
    model.train()

    # Initialize variables for collecting metrics from all batches
    loss_epoch = 0.0
    acc_epoch = 0.0
    offset = 0
    for i, sample in enumerate(train_loader):
        batch_size = sample['question'].size(0)

        # move data to GPU
        question = sample['question'].to(device)
        visual = sample['visual'].to(device)
        answer = sample['answer'].to(device)
        question_indexes = sample['question_id'] # keep in cpu
        mask = sample['mask'].to(device)

        # clear parameter gradients
        optimizer.zero_grad()
        # get output from model
        output = model(visual, question, mask)

        # compute loss
        loss = criterion(output.squeeze_(dim=-1), answer.float()) # cast to float because of BCEWithLogitsLoss 

        loss.backward()

        optimizer.step()

        # add running loss
        loss_epoch += loss.item()  
        # save probs and answers
        m = nn.Sigmoid()
        pred = m(output.data.cpu()) 
        # compute accuracy
        acc = metrics.batch_binary_accuracy((pred > 0.5).float().to(device), answer)
        results[offset:offset+batch_size,:] = torch.cat((question_indexes.view(batch_size, 1), torch.round(pred.view(batch_size,1))), dim=1)
        answers[offset:offset+batch_size] = torch.cat((answer.data.cpu().view(batch_size, 1), pred.view(batch_size,1)), dim=1)
        offset += batch_size
        acc_epoch += acc.item()

    # compute AUC and AP for current epoch
    auc, ap = metrics.compute_auc_ap(answers)
    metrics_dict = {'loss_train': loss_epoch/len(train_loader.dataset), 'auc_train': auc, 'ap_train': ap, 'acc_train': acc_epoch/len(train_loader.dataset)}
    logbook.log_metrics('train', metrics_dict, epoch)
    return metrics_dict


def validate_binary(val_loader, model, criterion, device, epoch, config, logbook, comet_exp=None):
    # tensor to save results
    results = torch.zeros((len(val_loader.dataset), 2), dtype=torch.int64) # to store question id, model's answer
    answers = torch.zeros(len(val_loader.dataset), 2) # store target answer, prob

    # set evaluation mode
    model.eval()

    # Initialize variables for collecting metrics from all batches
    loss_epoch = 0.0
    acc_epoch = 0.0
    offset = 0
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            batch_size = sample['question'].size(0)

            # move data to GPU
            question = sample['question'].to(device)
            visual = sample['visual'].to(device)
            answer = sample['answer'].to(device)
            question_indexes = sample['question_id'] # keep in cpu
            mask = sample['mask'].to(device)

            # get output from model
            output = model(visual, question, mask)

            # compute loss
            loss = criterion(output.squeeze_(dim=-1), answer.float())

            # save probs and answers
            m = nn.Sigmoid()
            pred = m(output.data.cpu())
            # compute accuracy
            acc = metrics.batch_binary_accuracy((pred > 0.5).float().to(device), answer)
            results[offset:offset+batch_size,0] = question_indexes
            results[offset:offset+batch_size,1] = torch.round(pred)
            answers[offset:offset+batch_size] = torch.cat((answer.data.cpu().view(batch_size, 1), pred.view(batch_size,1)), dim=1)
            offset += batch_size

            loss_epoch += loss.item()
            acc_epoch += acc.item()
            
    # compute AUC and AP for current epoch for all samples, using info in results
    auc, ap = metrics.compute_auc_ap(answers)
    metrics_dict = {'loss_val': loss_epoch/len(val_loader.dataset), 'auc_val': auc, 'ap_val': ap, 'acc_val': acc_epoch/len(val_loader.dataset)}
    if logbook is not None:
        logbook.log_metrics('val', metrics_dict, epoch)
    return metrics_dict, {'results': results, 'answers': answers}


def get_looper_functions(config):
    if config['num_answers'] == 2:
        train_fn = train_binary
        val_fn = validate_binary
    else:
        train_fn = train
        val_fn = validate
    return train_fn, val_fn