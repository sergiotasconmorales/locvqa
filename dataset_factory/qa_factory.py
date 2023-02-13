# Project:
#   Localized Questions in VQA
# Description:
#   QA pair creation
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern

import os
import numpy as np
import random
from PIL import Image
from os.path import join as jp
from skimage.measure import regionprops


def generate_random_window(h, w, min_side, max_side, prop, regions_in_subwindow=False, offset=0):
    p = random.random() # random number to decide whether random width or random height is sampled first
    if p >= 0.5:
        # sample width first
        random_w = random.randint(min_side, max_side)
        # to generate the random height I move back to the resized space so that the proportion is kept there instead of in the original space
        random_h = random.randint(max(min_side, round((1-prop)*random_w)), min(max_side, round((1+prop)*random_w)))
    else:
        # sample height first
        random_h = random.randint(min_side, max_side)
        # to generate the random width I move back to the resized space so that the proportion is kept there instead of in the original space
        random_w = random.randint(max(min_side, round((1-prop)*random_h)), min(max_side, round((1+prop)*random_h)))
    if regions_in_subwindow:
        # force windows to be in a specific sub-range of the original image so that in the resized space they are also within a sub-range (a centered sub-window)
        top_left_corner = (random.randint(offset, h - offset - random_h), random.randint(offset, w - offset - random_w))
    else:
        top_left_corner = (random.randint(0, h - random_h), random.randint(0, w - random_w))
    return top_left_corner, random_h, random_w

def convert_region_coords(top_left, window_h, window_w, h, w, new_side):
    """coords convertion function

    Parameters
    ----------
    top_left : tuple
        top left corner of the region in the original image size
    window_h : int
        height of the region in the original image size
    window_w : int
        width of the region in the original image size
    h : int
        height of the original image
    w : int
        width of the original image
    new_side : int
        new side of the resized image (square image)

    Returns
    -------
    tuple
        converted coords with the format ((top_left_y, top_left_x), h_, w_))
    """
    # convert to resized image coordinates
    Ry, Rx = new_side/h, new_side/w
    top_left_resized = (round(top_left[0]*Ry), round(top_left[1]*Rx))
    window_h_resized = round(window_h*Ry)
    window_w_resized = round(window_w*Rx)
    return top_left_resized, window_h_resized, window_w_resized

def generate_questions_about_regions(config, mask_gt, class_name, partial_qa_id, image_name, balanced=True, dataset='cholec'):
    """Generates questions about regions for the Cholec dataset

    Parameters
    ----------
    config : config dict
        _description_
    mask_gt : numpy array
        _description_
    class_name : str
        _description_
    partial_qa_id : int
        partial question id created from image index and class index
    image_name : str
        name of the image with extension
    balanced : bool, optional
        whether or not the dataset shoould be balanced, by default True
    dataset : str, optional
        name of the dataset, by default 'Cholec'

    Returns
    -------
    list
        List with QA pairs about random regions for current image
    """
    # get info from config
    num_regions = config['num_regions']
    min_window_side = config['min_window_side']
    max_window_side = config['max_window_side']
    threshold = config['threshold']
    proportion = config['proportion_deviation']

    # first, get number of pixels in the image
    num_pixels_img = mask_gt.shape[0] * mask_gt.shape[1]

    # now get number of 1s in mask
    if mask_gt.ndim > 2:
        num_pixels_mask = np.sum(mask_gt[:,:,0]) # take only one channel (all of them have the same amount - this info is in the Cholec info)
    else:
        num_pixels_mask = np.sum(mask_gt)

    if num_pixels_mask == 0:
        return [] # if there are no pixels in the mask, return empty list

    if dataset == 'sts2017' or dataset == 'insegcat': # due to the shape of the tools, use bounding box
        props = regionprops(mask_gt)
        # add the areas of all bounding boxes
        num_pixels_mask = 0
        for prop in props:
            minr, minc, maxr, maxc = prop.bbox
            num_pixels_mask += (maxr - minr)*(maxc - minc)

    # define number of regions. For now, I define it as a function that moves between 3 and num_regions. The idea is that if the GT region is too small
    # The idea here is that if the number of pixels in the mask is close to num_pixels_img/2, then num_regions should be produced. If the mask is too big or too small, then reduce the number of regions
    # so that small tissues are not redundant (regions on big tissues would correspond to different parts of the image, but small tissues would be just sampled several times)
    # This is described on pp. 23 of notebook 4
    if num_pixels_mask <= num_pixels_img/2:
        num_regions_recomputed = np.min([num_regions, np.max([config['min_regions'], int((2*(num_regions + (num_regions/2))/num_pixels_img)*num_pixels_mask)])])
        if num_regions_recomputed%2 != 0: # make even so that half of the questions can be answered with yes and half with no
            num_regions_recomputed += 1
    else:
        num_regions_recomputed = np.min([num_regions, np.max([config['min_regions'], int(-1*(2*(num_regions + (num_regions/2))/num_pixels_img)*(num_pixels_mask - num_pixels_img/2) + num_regions  + (num_regions/2) )])])
        if num_regions_recomputed%2 != 0: # make even so that half of the questions can be answered with yes and half with no
            num_regions_recomputed += 1


    qa_group = []
    i_region = 0 # region index for current image
    num_questions_yes = 0
    num_questions_no = 0
    budget = 100*num_regions_recomputed
    while num_questions_yes < round(num_regions_recomputed/2) or num_questions_no < round(num_regions_recomputed/2): # while not complete
        # generate randomly-sized region with random location
        top_left, window_h, window_w = generate_random_window(mask_gt.shape[0], mask_gt.shape[1], min_window_side, max_window_side, proportion, regions_in_subwindow=True, offset=config['window_offset'])
        # convert coordinates of the random region to the resized space
        top_left_resized, window_h_resized, window_w_resized = convert_region_coords(top_left, window_h, window_w, mask_gt.shape[0], mask_gt.shape[1], config['size'])
        # build mask array
        mask_region = np.zeros_like(mask_gt, dtype = np.uint8)
        mask_region[top_left[0]:top_left[0]+window_h, top_left[1]:top_left[1]+window_w] = 1 # * Important: to be used like this in dataset class to create the mask, but setting it to 255

        num_pixels_in_region = np.count_nonzero(mask_gt*mask_region)

        if (num_pixels_in_region >= threshold) and num_questions_yes < round(num_regions_recomputed/2): # if answer is yes and i haven't reached the maximum number of positive questions
            answer = 'yes'
            question_linked_to_region_mask = ('is there ' + class_name + ' in this region?').lower()
            question_mentioning_region = ('is there ' + class_name + ' in the region with top left corner at (' + str(top_left_resized[0]) + ', ' + str(top_left_resized[1]) + ') and height ' + str(window_h_resized) + ' and width ' + str(window_w_resized) + '?').lower()
            qa_group.append({
                                'image_name': image_name,
                                'question': question_linked_to_region_mask,
                                'question_alt': question_mentioning_region,
                                'question_id': int(str(i_region+1).zfill(3) + partial_qa_id),
                                'question_type': 'region',
                                'mask_coords': (top_left_resized, window_h_resized, window_w_resized),
                                'mask_coords_orig': (top_left, window_h, window_w), # save coords in original space just in case
                                'answer': answer,
                                'mask_size': (config['size'], config['size']),
                                'mask_size_orig': mask_region.shape, # original shape
                                'question_object': class_name
            })
            num_questions_yes += 1
            i_region += 1

        elif num_pixels_in_region == 0 and num_questions_no < round(num_regions_recomputed/2): # if answer is no and i haven't reached the maximum number of negative questions
            answer = 'no'
            
            question_linked_to_region_mask = ('is there ' + class_name + ' in this region?').lower()
            question_mentioning_region = ('is there ' + class_name + ' in the region with top left corner at (' + str(top_left_resized[0]) + ', ' + str(top_left_resized[1]) + ') and height ' + str(window_h_resized) + ' and width ' + str(window_w_resized) + '?').lower()

            qa_group.append({
                                'image_name': image_name,
                                'question': question_linked_to_region_mask,
                                'question_alt': question_mentioning_region,
                                'question_id': int(str(i_region+1).zfill(3) + partial_qa_id),
                                'question_type': 'region',
                                'mask_coords': (top_left_resized, window_h_resized, window_w_resized),
                                'mask_coords_orig': (top_left, window_h, window_w),
                                'answer': answer,
                                'mask_size': (config['size'], config['size']),
                                'mask_size_orig': mask_region.shape,
                                'question_object': class_name
            })
            num_questions_no += 1
            i_region += 1
        else:
            budget -= 1

        if budget == 0: # if I can't find a region that satisfies the conditions, I stop
            print('WARNING: budget exceeded for image ' + image_name + ' and class ' + class_name + '. Skipping class')
            return []


    assert num_questions_no == num_questions_yes # sanity check: check balance
    return qa_group


def generate_questions_about_image(config, labels, mask_code, image_name, h, w, img_idx):
    list_classes_int = set(mask_code.keys())
    labels_in_image = set(labels)

    code2class = {v: k for k, v in mask_code.items()}

    to_choose_from = list_classes_int - labels_in_image # classes that are not in the image (negative questions)

    num_yes = 0
    num_no = 0
    qa_group = []

    # first, create positive questions
    labels_in_image_text = [mask_code[l] for l in labels_in_image]
    idx = 0
    for class_name in labels_in_image_text:
        question_linked_to_region_mask = ('is there ' + class_name + ' in this image?').lower()
        question_mentioning_region = ('is there ' + class_name + ' in the region with top left corner at (' + str(0) + ', ' + str(0) + ') and height ' + str(config['size']) + 'and width ' + str(config['size']) + '?').lower()
        qa_group.append({
                            'image_name': image_name,
                            'question': question_linked_to_region_mask,
                            'question_alt': question_mentioning_region,
                            'question_id': int(str(idx+1).zfill(2) + '2' + str(code2class[class_name]).zfill(2) + img_idx),
                            'question_type': 'whole',
                            'mask_coords': ((0,0), config['size'], config['size']),
                            'mask_coords_orig': ((0,0), h, w),
                            'answer': 'yes',
                            'mask_size': (config['size'], config['size']),
                            'mask_size_orig': (h, w),
                            'question_object': class_name
        })
        num_yes += 1
        idx += 1

    # then, create negative questions
    for i in range(len(labels_in_image)):
        # choose a random class that is not in the image
        class_name = mask_code[np.random.choice(list(to_choose_from))]
        question_linked_to_region_mask = ('is there ' + class_name + ' in this image?').lower()
        question_mentioning_region = ('is there ' + class_name + ' in the region with top left corner at (' + str(0) + ', ' + str(0) + ') and height ' + str(h) + 'and width ' + str(w) + '?').lower()
        qa_group.append({
                            'image_name': image_name,
                            'question': question_linked_to_region_mask,
                            'question_alt': question_mentioning_region,
                            'question_id': int(str(idx+1).zfill(2) + '2' + str(code2class[class_name]).zfill(2) + img_idx),
                            'question_type': 'whole',
                            'mask_coords': ((0,0), config['size'], config['size']),
                            'mask_coords_orig': ((0,0), h, w),
                            'answer': 'no',
                            'mask_size': (config['size'], config['size']),
                            'mask_size_orig': (h, w),
                            'question_object': class_name
        }) 
        num_no += 1
        idx += 1

    assert num_yes == num_no # sanity check: check balance
    return qa_group

