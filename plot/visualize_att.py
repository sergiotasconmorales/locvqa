# Project:
#   Localized Questions in VQA
# Description:
#  Guided attention visualization
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from os.path import join as jp
from core.datasets.visual import default_inverse_transform as dit
import cv2

m = nn.Sigmoid()
sm = nn.Softmax(dim=2)

att = {}
def get_att_map(name):
    def hook(model, input, output):
        att[name] = output.detach()
    return hook 

def plot_attention_maps(model_name, model, visual, question, mask, answer, vocab_words, path_logs, question_indexes, vocab_answers):
    # if data parallel, get the model
    if model.__class__.__name__ == 'DataParallel':
        model = model.module
    if model_name == 'VQA_MaskRegion':
        model.attention_mechanism.conv2.register_forward_hook(get_att_map('attention_mechanism.conv2'))
        output = model(visual, question, mask)
        pred = (m(output.data.cpu())>0.5).to(torch.int64)
        k = att['attention_mechanism.conv2'] # size 64, 2, 14, 14 for sts
        h = k.clone()
        h = h.view(output.shape[0],2,14*14) 
        h_out = sm(h)
        g_out = h_out.view(output.shape[0],2,14,14)
        for i_s in range(g_out.shape[0]): # for every element of the batch
            image = dit()(visual[i_s]).permute(1,2,0).cpu().numpy()
            plt.ioff()
            f, ax = plt.subplots(1, 3)
            f.tight_layout()
            ax[0].imshow(image)
            ax[0].axis('off')
            question_words_encoded = [vocab_words[question[i_s, i].item()] for i in range(question.shape[1]) if question[i_s, i].item()!= 0]
            question_text = ' '.join(question_words_encoded)
            f.suptitle(question_text + "\n GT: " + str(vocab_answers[answer[i_s].item()]) + ', Pred: ' + str(vocab_answers[pred[i_s].item()]))
            ax[0].set_title('Image')
            #ax[0].set_title(question_text + "\n, GT: " + str(vocab_answers[answer[i_s].item()]))
            if pred[i_s].item() == answer[i_s].item():
                f.set_facecolor("green")
            else:
                f.set_facecolor("r")
            for i_glimpse in range(g_out.shape[1]): # for every glimpse
                img1 = g_out[i_s, i_glimpse, :, :].cpu().numpy()
                heatmap = cv2.resize(img1, (image.shape[1], image.shape[0]))
                heatmap = (heatmap - np.min(heatmap))/(np.max(heatmap) - np.min(heatmap))
                heatmap = np.uint8(255*heatmap)
                #heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                norm = plt.Normalize()
                heatmap = plt.cm.jet(norm(heatmap))
                superimposed = heatmap[:,:,:3] * 0.4 + image*mask[i_s].permute(1,2,0).cpu().numpy()
                superimposed = 255*(superimposed - np.min(superimposed))/(np.max(superimposed) - np.min(superimposed))
                ax[i_glimpse+1].imshow(superimposed.astype(np.uint8))
                ax[i_glimpse+1].axis('off')
                ax[i_glimpse+1].set_title("Glimpse " + str(i_glimpse+1))
            plt.savefig(jp(path_logs, 'att_maps', str(question_indexes[i_s].item()) + '.png') ,bbox_inches='tight')
            plt.close()
    elif model_name == 'vQA_IgnoreMask':
        pass
    elif model_name == 'VQA_LocalizedAttention':
        model.attention_mechanism.conv2.register_forward_hook(get_att_map('attention_mechanism.conv2'))
        output = model(visual, question, mask)
        pred = (m(output.data.cpu())>0.5).to(torch.int64)
        k = att['attention_mechanism.conv2'] # size 64, 2, 14, 14 for sts
        h = k.clone()
        h = h.view(output.shape[0],2,14*14) 
        h_out = sm(h)
        g_out = h_out.view(output.shape[0],2,14,14)
        for i_s in range(g_out.shape[0]): # for every element of the batch
            image = dit()(visual[i_s]).permute(1,2,0).cpu().numpy()
            f, ax = plt.subplots(1, 3)
            f.tight_layout()
            ax[0].imshow(image)
            if not np.count_nonzero(mask[i_s].cpu().numpy()) == mask[i_s].shape[-1]*mask[i_s].shape[-2]:
                masked = np.ma.masked_where(mask[i_s].permute(1,2,0).cpu().numpy() ==0, mask[i_s].permute(1,2,0).cpu().numpy())
                ax[0].imshow(masked, 'jet', interpolation='none', alpha=0.5)
            ax[0].axis('off')
            question_words_encoded = [vocab_words[question[i_s, i].item()] for i in range(question.shape[1]) if question[i_s, i].item()!= 0]
            question_text = ' '.join(question_words_encoded)
            f.suptitle(question_text + "\n GT: " + str(vocab_answers[answer[i_s].item()]) + ', Pred: ' + str(vocab_answers[pred[i_s].item()]))
            ax[0].set_title('Image')
            #ax[0].set_title(question_text + "\n, GT: " + str(vocab_answers[answer[i_s].item()]))
            if pred[i_s].item() == answer[i_s].item():
                f.set_facecolor("green")
            else:
                f.set_facecolor("r")
            for i_glimpse in range(g_out.shape[1]): # for every glimpse
                img1 = g_out[i_s, i_glimpse, :, :].cpu().numpy()
                heatmap = cv2.resize(img1, (image.shape[1], image.shape[0]))
                heatmap = (heatmap - np.min(heatmap))/(np.max(heatmap) - np.min(heatmap))
                heatmap = np.uint8(255*heatmap)
                #heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                norm = plt.Normalize()
                heatmap = plt.cm.jet(norm(heatmap))
                superimposed = heatmap[:,:,:3] * 0.4 + image
                superimposed = 255*(superimposed - np.min(superimposed))/(np.max(superimposed) - np.min(superimposed))
                ax[i_glimpse+1].imshow(superimposed.astype(np.uint8))
                ax[i_glimpse+1].axis('off')
                ax[i_glimpse+1].set_title("Glimpse " + str(i_glimpse+1))
            plt.savefig(jp(path_logs, 'att_maps', str(question_indexes[i_s].item()) + '.png') ,bbox_inches='tight')
            plt.close()
    elif model_name == 'VQARS_4':
        pass
    else:
        raise ValueError('Model not supported') 

def plot_attention_single(model_name, model, visual, question, mask, path_output, question_id, case, idx2ans, mask_as_text=False, tag='404'):
    if model.__class__.__name__ == 'DataParallel':
        model = model.module
    if model_name == 'VQA_MaskRegion':
        model.attention_mechanism.conv2.register_forward_hook(get_att_map('attention_mechanism.conv2'))
        output = model(visual, question, mask)
        pred = (m(output.data.cpu())>0.5).to(torch.int64)
        #pred = torch.argmax(output, dim=1)
        k = att['attention_mechanism.conv2'] # size 64, 2, 14, 14 for sts
        h = k.clone()
        h = h.view(output.shape[0],2,14*14) 
        h_out = sm(h)
        g_out = h_out.view(output.shape[0],2,14,14)
        for i_s in range(g_out.shape[0]): # for every element of the batch
            image = dit()(visual[i_s]).permute(1,2,0).cpu().numpy()
            answer = idx2ans[pred[i_s].item()]
            for i_glimpse in range(g_out.shape[1]): # for every glimpse
                plt.ioff()
                img1 = g_out[i_s, i_glimpse, :, :].cpu().numpy()
                heatmap = cv2.resize(img1, (image.shape[1], image.shape[0]))
                heatmap = (heatmap - np.min(heatmap))/(np.max(heatmap) - np.min(heatmap))
                heatmap = np.uint8(255*heatmap)
                norm = plt.Normalize()
                heatmap = plt.cm.jet(norm(heatmap))
                superimposed = heatmap[:,:,:3] * 0.4 + image*mask[i_s].permute(1,2,0).cpu().numpy()
                superimposed = 255*(superimposed - np.min(superimposed))/(np.max(superimposed) - np.min(superimposed))
                # save image
                if i_glimpse == 0: # only first glimpse
                    plt.imsave(jp(path_output, tag, case, str(question_id[i_s].item()) + '_' + model_name + '_g' + str(i_glimpse) + '_'+ str(answer) +'.png') ,superimposed.astype(np.uint8))
    elif model_name == 'VQA_LocalizedAttention':
        model.attention_mechanism.conv2.register_forward_hook(get_att_map('attention_mechanism.conv2'))
        output = model(visual, question, mask)
        pred = (m(output.data.cpu())>0.5).to(torch.int64)
        #pred = torch.argmax(output, dim=1)
        k = att['attention_mechanism.conv2'] # size 64, 2, 14, 14 for sts
        h = k.clone()
        h = h.view(output.shape[0],2,14*14) 
        h_out = sm(h)
        g_out = h_out.view(output.shape[0],2,14,14)
        for i_s in range(g_out.shape[0]): # for every element of the batch
            image = dit()(visual[i_s]).permute(1,2,0).cpu().numpy()
            answer = idx2ans[pred[i_s].item()]
            for i_glimpse in range(g_out.shape[1]): # for every glimpse
                plt.ioff()
                img1 = g_out[i_s, i_glimpse, :, :].cpu().numpy()
                heatmap = cv2.resize(img1, (image.shape[1], image.shape[0]))
                heatmap = (heatmap - np.min(heatmap))/(np.max(heatmap) - np.min(heatmap))
                heatmap = np.uint8(255*heatmap)
                norm = plt.Normalize()
                heatmap = plt.cm.jet(norm(heatmap))
                superimposed = heatmap[:,:,:3]*mask[i_s].permute(1,2,0).cpu().numpy() * 0.4 + image
                superimposed = 255*(superimposed - np.min(superimposed))/(np.max(superimposed) - np.min(superimposed))
                # save image
                if i_glimpse == 0: # only first glimpse
                    plt.imsave(jp(path_output,tag, case, str(question_id[i_s].item()) + '_' + model_name + '_g' + str(i_glimpse) + '_'+ str(answer) +'.png') ,superimposed.astype(np.uint8))
    elif model_name == 'VQA_IgnoreMask' or (model_name == 'VQA_Base'):
        model.attention_mechanism.conv2.register_forward_hook(get_att_map('attention_mechanism.conv2'))
        output = model(visual, question, mask)
        pred = (m(output.data.cpu())>0.5).to(torch.int64)
        #pred = torch.argmax(output, dim=1)
        k = att['attention_mechanism.conv2'] # size 64, 2, 14, 14 for sts
        h = k.clone()
        h = h.view(output.shape[0],2,14*14) 
        h_out = sm(h)
        g_out = h_out.view(output.shape[0],2,14,14)
        for i_s in range(g_out.shape[0]): # for every element of the batch
            image = dit()(visual[i_s]).permute(1,2,0).cpu().numpy()
            answer = idx2ans[pred[i_s].item()]
            for i_glimpse in range(g_out.shape[1]): # for every glimpse
                plt.ioff()
                img1 = g_out[i_s, i_glimpse, :, :].cpu().numpy()
                heatmap = cv2.resize(img1, (image.shape[1], image.shape[0]))
                heatmap = (heatmap - np.min(heatmap))/(np.max(heatmap) - np.min(heatmap))
                heatmap = np.uint8(255*heatmap)
                norm = plt.Normalize()
                heatmap = plt.cm.jet(norm(heatmap))
                superimposed = heatmap[:,:,:3] * 0.4 + image
                superimposed = 255*(superimposed - np.min(superimposed))/(np.max(superimposed) - np.min(superimposed))
                # save image
                if i_glimpse == 0: # only first glimpse
                    plt.imsave(jp(path_output, tag, case, str(question_id[i_s].item()) + '_' + model_name + '_g' + str(i_glimpse) + '_'+ str(answer) +'.png') ,superimposed.astype(np.uint8))

    elif model_name == 'VQA_Base' and not mask_as_text:
        pass