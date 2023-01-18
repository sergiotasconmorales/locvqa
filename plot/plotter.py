# Project:
#   Localized Questions in VQA
# Description:
#   Plotting functions
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern

from matplotlib import pyplot as plt
import numpy as np

def show_image(img, title=None, cmap=None):
    plt.imshow(img, cmap=cmap)
    if title is not None:
        plt.title(title)
    plt.show()

def plot_learning_curve(metric_dict_train, metric_dict_val, metric_name, x_label='epoch', title="Learning curve", save=False, path=None):
    """ Input dictionaries are expected to have epoch indexes (string) as keys and floats as values"""
    fig = plt.figure()
    if metric_name == 'loss':
        top_val = max(max(list(metric_dict_train.values())), max(list(metric_dict_val.values()))) 
    else:
        top_val = 1.0
        metric_name = metric_name.upper()
        
    # plot train metrics
    plt.plot([int(e) for e in metric_dict_train.keys()], list(metric_dict_train.values()), label=metric_name + ' train', linewidth=2, color='orange')
    # plot val metrics
    plt.plot([int(e) for e in metric_dict_val.keys()], list(metric_dict_val.values()), label=metric_name + ' val', linewidth=2, color='blue')
    plt.xticks([int(e) for e in metric_dict_train.keys()])
    plt.grid()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylim((0, top_val))
    plt.ylabel(metric_name)
    plt.legend()
    if save:
        if path is not None:
            plt.savefig(jp(path, metric_name + '.png'), dpi=300)
        else:
            raise ValueError


def plot_roc_prc(roc, auc, prc, ap, title='ROC and PRC plots', save=True, path=None, suffix=''):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    f.suptitle(title)
    # plot PRC
    ax1.plot(prc[1], prc[0], label = "PRC , AP: " + "{:.3f}".format(ap))
    #ax1.plot([0, 1], [no_skill, no_skill], linestyle='--', color = colors[k], label='No Skill')
    ax1.set_xlabel("recall")
    ax1.set_ylabel("precision")
    ax1.grid()
    ax1.legend()

    # plot ROC
    ax2.plot(roc[0], roc[1],label = "ROC, AUC: " + "{:.3f}".format(auc))
    #ax2.plot(fpr_dumb, tpr_dumb, linestyle="--", color = "gray", label="No Skill")
    ax2.set_xlabel("fpr")
    ax2.set_ylabel("tpr")
    ax2.grid()
    ax2.legend()

    if save and path is not None:
        plt.savefig(jp(path, 'ROC_PRC_' + suffix + '.png'), dpi=300)


def overlay_mask(img, mask, gt, save= False, path_without_ext=None, alpha = 0.7, fig = None, ax = None):
    masked = np.ma.masked_where(mask ==0, mask)
    gt = np.ma.masked_where(gt==0, gt)
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    ax.imshow(img, 'gray', interpolation='none')
    ax.imshow(masked, 'jet', interpolation='none', alpha=alpha)
    ax.imshow(gt, 'pink', interpolation='none', alpha=alpha)
    #fig.set_facecolor("black")
    fig.tight_layout()
    ax.axis('off')
    if save:
        plt.savefig(path_without_ext + '.png', bbox_inches='tight')
    else:
        plt.show()