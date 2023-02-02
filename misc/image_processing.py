# Project:
#   Localized Questions in VQA
# Description:
#   Image processing functions
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern

from PIL import Image
import numpy as np




def resize_and_save(path_in, path_out, size = 448):
    """Normalization function
    """
    im = Image.open(path_in)

    im_resized = im.resize((size, size), Image.ANTIALIAS)

    im_resized.save(path_out)