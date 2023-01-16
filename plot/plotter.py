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