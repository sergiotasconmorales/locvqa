# Project:
#   Localized Questions in VQA
# Description:
#   Class implementations for the creation of region-based (a.k.a. localized) VQA datasets
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern

import os
import shutil
from tqdm import tqdm
from misc import dirs, printer, io, image_processing
from plot import plotter
import random
from os.path import join as jp
import numpy as np
from . import qa_factory
from .coco.PythonAPI.pycocotools.coco import COCO
from skimage.measure import regionprops
import copy

PENDING = 0

class RegionsDatasetCreator(object):
    # define init
    def __init__(self, config):
        self.config = config
        self.dataset_name = config['dataset']
        self.path_data = config['path_data']
        self.path_output = config['path_output']
        self.path_images_orig = os.path.join(self.path_output, 'original_images')
        self.path_images_output = os.path.join(self.path_output, 'images')
        self.path_qa_output = os.path.join(self.path_output, 'qa')
        os.makedirs(self.path_output, exist_ok=True)
        dirs.create_folders_within_folder(self.path_output, ['original_images', 'images', 'qa'])
        self.num_regions = config['num_regions']
        self.built_success = False

    # generic method to print details of the dataset
    def print_details(self):
        print('The', self.dataset_name, 'dataset will be created.')
        print('Result will be stored at: {}'.format(self.path_output))

    # Build dataset
    def build(self):
        raise NotImplementedError

    def created_successfully(self):
        if self.built_success:
            print('Dataset was created successfully.')
        else:
            print('Dataset was not created successfully.')

    def process_dataset(self):
        self.built_success = True


class Cholec(RegionsDatasetCreator):
    # Class to create the localized VQA dataset from the CholecSeg8k dataset 

    # call parent class
    def __init__(self, config):
        super().__init__(config)
        self.mask_code = {  50: 'black background', 
                            11: 'abdominal wall',
                            21: 'liver',
                            13: 'gastrointestinal tract',
                            12: 'fat',
                            31: 'grasper',
                            23: 'connective tissue',
                            24: 'blood', 
                            25: 'cystic duct',
                            32: 'l-hook electrocautery',
                            22: 'gallbladder',
                            33: 'hepatic vein',
                            5: 'liver ligament'}

    # overwrite build method for this dataset
    def build(self):
        # Steps
        # 1. Organize data
        self.preprocess_data()
        # 2. Create dataset
        self.create_dataset()
        # 3. Process dataset
        self.process_dataset()

    def preprocess_data(self):
        # here I need to split the data by video, not by frames because frames are too similar. Since there are 8080 frames in total, test videos should contain approx. 1616 frames.
        # Then divide trainval with 20% for val. Split was made taking this into account but not randomly because the amount of frames per vide can vary considerably. Pp. 19 of Notebook 4. 
        split = {'train': [1,9,12,18,20,24,26,27,35,52], 'val': [25,43,55], 'test': [17,28,37,48]}
        # now put images together in a new folder in path_output
        # create train, val and test folders in path_output
        for subset, video_idx in tqdm(split.items(), desc='Organizing data', colour='blue'):
            os.makedirs(os.path.join(self.path_images_orig, subset), exist_ok=True)
            # copy images from selected videos to folders
            for idx in video_idx:
                # list sub-folders
                folder_segments = os.listdir(os.path.join(self.path_data, 'video' + str(idx).zfill(2)))
                for folder_segment in folder_segments:
                    # list images
                    images = [e for e in os.listdir(os.path.join(self.path_data, 'video' + str(idx).zfill(2), folder_segment)) if 'color_mask' not in e and 'endo_mask.png' not in e]
                    for image in images:
                        image_name = str(idx).zfill(2) + '_' + image.split('_')[1] + '_' + ''.join(image.split('_')[2:])
                        if 'endowatershedmask' in image_name:
                            image_name = image_name.replace('endowatershedmask', 'mask')
                        if not os.path.exists(os.path.join(self.path_images_orig, subset, image_name)) or self.config['overwrite_img']:
                            shutil.copy(os.path.join(self.path_data, 'video' + str(idx).zfill(2), folder_segment, image), os.path.join(self.path_images_orig, subset, image_name))
                        else:
                            print("Skipping original image", image_name, "File exists and overwrite_img is set to False")
        # now resize images from path_images_orig and store them in path_images_output
        for subset in ['train', 'val', 'test']:
            os.makedirs(jp(self.path_images_output, subset), exist_ok=True)
            images = [e for e in os.listdir(jp(self.path_images_orig, subset)) if 'mask' not in e]
            for image in tqdm(images, desc='Resizing images ' + subset, colour='blue'):
                if not os.path.exists(jp(self.path_images_output, subset, image)) or self.config['overwrite_img']:
                    image_processing.resize_and_save(jp(self.path_images_orig, subset, image), jp(self.path_images_output, subset, image), size = self.config['size'])
                else:
                    print("Skipping resizing of image", image, "File exists and overwrite_img is set to False")


    def create_dataset(self):
        printer.print_section('Creating QA pairs')
        # Now generate questions about regions and about whole images for each split
        for subset in ['train', 'val', 'test']:
            if os.path.exists(jp(self.path_qa_output, subset + 'qa.json')) and not self.config['overwrite_qa']:
                print("Skipping qa file", subset + '_qa.json', "File exists and overwrite_qa is set to False")
                continue # if qa file exists and no overwrite is required, skip

            qa = [] # to save QA entries

            # list all images in current subset (original size)
            images = [e for e in dirs.list_files(jp(self.path_images_orig, subset)) if 'mask' not in e]
            # for each image, generate a set of questions about random regions, based on the labels that are present in the image (balanced)
            print(subset, 'set...')
            for i_image, image in enumerate(tqdm((images), desc='Creating QA pairs for ' + subset, colour='blue')):
                # get mask
                mask = image.replace('endo', 'mask')
                # read image
                img = io.read_image(jp(self.path_images_orig, subset, image))
                # read mask
                mask = io.read_image(jp(self.path_images_orig, subset, mask))

                # get unique labels in mask
                labels = [e for e in list(set(mask.flatten())) if e in set(self.mask_code.keys())]

                # for each label, generate binary mask and then generate pairs of questions about regions
                for l in labels:
                    # create binary mask
                    mask_bin = np.zeros_like(mask)
                    mask_bin[mask == l] = 1
                    # generate pairs of questions
                    partial_qa_id = '1' +  str(l).zfill(2) + str(i_image+1).zfill(4)
                    qa_pairs = qa_factory.cholec_generate_questions_about_regions(self.config, mask_bin, self.mask_code[l], partial_qa_id, image, balanced=True)
                    qa.extend(qa_pairs)

                qa_pairs = qa_factory.cholec_generate_questions_about_image(self.config, labels, self.mask_code, image, img.shape[0], img.shape[1], str(i_image+1).zfill(4))
                qa.extend(qa_pairs)

            io.save_json(qa, jp(self.path_qa_output, subset + '_qa.json'))

class Insegcat(RegionsDatasetCreator):
    # call parent class
    def __init__(self, config):
        super().__init__(config)
        self.subsets = ['training', 'validation', 'test']

    # overwrite build method for this dataset
    def build(self):
        # Steps
        # 1. Organize data
        self.preprocess_data()
        # 2. Create dataset
        self.create_dataset()
        # 3. Process dataset
        self.process_dataset()

    # data preprocessing and organization
    def preprocess_data(self):
        # copy images from original folder to new folder. Also, generate mask files from annotations files using COCO API
        for subset in tqdm(self.subsets, desc='Organizing data', colour='blue'):
            os.makedirs(jp(self.path_images_orig, subset), exist_ok=True)
            images_subset = os.listdir(jp(self.path_data, subset))
            # create COCO object for current subset
            coco = COCO(jp(self.path_data, subset + '.json'))
            imgs = coco.imgs
            name2id = {imgs[e]['file_name'] : e for e in imgs.keys()}
            self.mask_code = {coco.cats[e]['id'] : coco.cats[e]['name'] for e in coco.cats.keys()}
            for image in tqdm(images_subset, desc='Copying images ' + subset, colour='blue'):
                if not os.path.exists(jp(self.path_images_orig, subset, image)) or self.config['overwrite_img']:
                    shutil.copy(jp(self.path_data, subset, image), jp(self.path_images_orig, subset, image))
                else:
                    print("Skipping original image", image, "File exists and overwrite_img is set to False")
                # now deal with the masks
                # get image id
                image_id = name2id[image]
                # get annotations for current image
                anns = coco.getAnnIds(imgIds=image_id)
                anns = coco.loadAnns(anns)
                # create mask
                mask = np.zeros((coco.imgs[image_id]['height'], coco.imgs[image_id]['width']))
                for ann in anns:
                    mask += coco.annToMask(ann)*ann['category_id']
                # save mask
                if not os.path.exists(jp(self.path_images_orig, subset, image.replace('.png', '_mask.png'))) or self.config['overwrite_img']:
                    io.save_image(mask.astype(np.uint8), jp(self.path_images_orig, subset, image.replace('.png', '_mask.png')))
                else:
                    print("Skipping mask image", image, "File exists or overwrite_img is set to False")

        # now resize images from path_images_orig and store them in path_images_output
        for subset in tqdm(self.subsets, desc='Resizing images', colour='yellow'):
            os.makedirs(jp(self.path_images_output, subset), exist_ok=True)
            images = [e for e in os.listdir(jp(self.path_images_orig, subset)) if 'mask' not in e]
            for image in tqdm(images, desc='Resizing images ' + subset, colour='blue'):
                if not os.path.exists(jp(self.path_images_output, subset, image)) or self.config['overwrite_img']:
                    image_processing.resize_and_save(jp(self.path_images_orig, subset, image), jp(self.path_images_output, subset, image), size = self.config['size'])
                else:
                    print("Skipping resizing of image", image, "File exists and overwrite_img is set to False") 
                
    # create dataset
    def create_dataset(self):
        printer.print_section('Creating QA pairs')
        # Now generate questions about regions and about whole images for each split
        for subset in self.subsets:
            if os.path.exists(jp(self.path_qa_output, subset + 'qa.json')) and not self.config['overwrite_qa']:
                print("Skipping qa file", subset + '_qa.json', "File exists and overwrite_qa is set to False")
                continue # if qa file exists and no overwrite is required, skip

            qa = [] # to save QA entries

            # list all images in current subset (original size)
            images = [e for e in dirs.list_files(jp(self.path_images_orig, subset)) if 'mask' not in e]
            # for each image, generate a set of questions about random regions, based on the labels that are present in the image (balanced)
            print(subset, 'set...')
            for i_image, image in enumerate(tqdm((images), desc='Creating QA pairs for ' + subset, colour='blue')):
                # get mask
                mask = image.replace('.png', '_mask.png')
                # read image
                img = io.read_image(jp(self.path_images_orig, subset, image))
                # read mask
                mask = io.read_image(jp(self.path_images_orig, subset, mask))

                # get unique labels in mask
                labels = [e for e in list(set(mask.flatten())) if e in set(self.mask_code.keys())]

                # for each label, generate binary mask and then generate pairs of questions about regions
                for l in labels:
                    # create binary mask
                    mask_bin = np.zeros_like(mask)
                    mask_bin[mask == l] = 1
                    # generate pairs of questions
                    partial_qa_id = '1' +  str(l).zfill(2) + str(i_image+1).zfill(4)
                    qa_pairs = qa_factory.generate_questions_about_regions(self.config, mask_bin, self.mask_code[l], partial_qa_id, image, balanced=True, dataset = 'insegcat')
                    qa.extend(qa_pairs)

                # no questions about whole images for this dataset
                #qa_pairs = qa_factory.cholec_generate_questions_about_image(self.config, labels, self.mask_code, image, img.shape[0], img.shape[1], str(i_image+1).zfill(4))
                #qa.extend(qa_pairs)

            io.save_json(qa, jp(self.path_qa_output, subset + '_qa.json'))


class STS2017(RegionsDatasetCreator):
    # Class to create the localized VQA dataset from the surgical tools segmentation challenge 2017 dataset 

    # call parent class
    def __init__(self, config):
        super().__init__(config)
        self.color_code = {0 : [0, 0, 0], 1 : [0, 0, 255], 2 : [0, 255, 0], 3 : [255, 0, 0], 4 : [255, 255, 0], 5 : [255, 0, 255], 6 : [0, 255, 255], 7 : [255, 255, 255]}

    # overwrite build method for this dataset
    def build(self):
        # Steps
        # 1. Organize data
        self.preprocess_data()
        # 2. Create dataset
        self.create_dataset()
        # 3. Process dataset
        self.process_dataset()

    # data preprocessing and organization
    def preprocess_data(self):
        path_train = jp(self.path_data, 'train')
        # read mappings
        self.mapping_tool2index = io.read_json(jp(path_train, 'instrument_type_mapping.json'))
        self.mapping_index2tool = {v : k for k,v in self.mapping_tool2index.items()}
        self.mapping_tool_part = io.read_json(jp(path_train, 'mappings.json'))
        folder2tool = {'_'.join(k.split()) + '_labels' : v for k,v in self.mapping_tool2index.items()}
        train_sequences = dirs.list_folders(path_train)
        
        for seq in tqdm(train_sequences, desc='Converting train GT format to test format'):
            # create folder gt within folder
            os.makedirs(jp(path_train, seq, 'gt'), exist_ok=True)
            
            # do the following only if folder BinarySegmentation does not exist 
            if not os.path.exists(jp(path_train, seq, 'gt', 'BinarySegmentation')) or len(os.listdir(jp(path_train, seq, 'gt', 'BinarySegmentation'))) == 0 or self.config['overwrite_img']:
                dirs.create_folders_within_folder(jp(path_train, seq, 'gt'), ['BinarySegmentation'])
                # save paths in variables for easy handling
                binseg_path = jp(path_train, seq, 'gt', 'BinarySegmentation') # contains union of segmentation about tools (no Other)
                left_frame_path = jp(path_train, seq, 'left_frames')
                left_frame_images = dirs.list_files(left_frame_path)
                gt_orig_path = jp(path_train, seq, 'ground_truth')
                # list folders in gt_orig_path
                tools = [e for e in dirs.list_folders(gt_orig_path)]
                # for each image in left_frame_images, generate a binary image from all masks in tools by applying OR
                for img in left_frame_images:
                    img_bin = np.zeros_like(io.read_image(jp(left_frame_path, img))[:,:,0]) # only one channel because binary shouldn't have 3 channels
                    for tool in tools:
                        mask_tool = io.read_image(jp(gt_orig_path, tool, img))
                        if mask_tool.ndim == 3: # some gt images seem to have 3 channels
                            mask_tool = mask_tool.sum(axis=-1)
                        img_bin = np.logical_or(img_bin, mask_tool)
                    io.save_image((img_bin.astype(np.uint8))*255, jp(binseg_path, img))

            if not os.path.exists(jp(path_train, seq, 'gt', 'TypeSegmentation')) or len(os.listdir(jp(path_train, seq, 'gt', 'TypeSegmentation'))) == 0 or self.config['overwrite_img']:
                # here about type segmentation. I.e. depending on the name of the folders in ground_truth, generate tool type
                dirs.create_folders_within_folder(jp(path_train, seq, 'gt'), ['TypeSegmentation', 'TypeSegmentationRescaled'])
                # save paths in variables for easy handling
                typeseg_path = jp(path_train, seq, 'gt', 'TypeSegmentation')
                typesegrescaled_path = jp(path_train, seq, 'gt', 'TypeSegmentationRescaled')
                left_frame_path = jp(path_train, seq, 'left_frames')
                left_frame_images = dirs.list_files(left_frame_path)
                gt_orig_path = jp(path_train, seq, 'ground_truth')
                # list folders in gt_orig_path
                tools = [e for e in dirs.list_folders(gt_orig_path)]                
                for img in left_frame_images:
                    img_type = np.zeros_like(io.read_image(jp(left_frame_path, img))[:,:,0]) # only one channel because binary shouldn't have 3 channels
                    img_type_rescaled = np.zeros_like(io.read_image(jp(left_frame_path, img))) 
                    for tool in tools: # for each folder
                        mask_tool = io.read_image(jp(gt_orig_path, tool, img))
                        if mask_tool.ndim == 3: # some gt images seem to have 3 channels
                            mask_tool = mask_tool.sum(axis=-1)
                        if 'Right' in tool or 'Left' in tool: # if folder name contains Right or Left, remove it
                            tool = '_'.join(tool.split('_')[1:])
                        img_type[mask_tool>0] = folder2tool[tool]
                        img_type_rescaled[mask_tool>0] = self.color_code[folder2tool[tool]]
                    io.save_image((img_type.astype(np.uint8)), jp(typeseg_path, img))
                    io.save_image((img_type_rescaled.astype(np.uint8)), jp(typesegrescaled_path, img))

        # now copy images to self.path_images_orig, separating train, val test. Crop them and then save them
        # define data for val. Take some whole sequences or randomly sample every sequence? I will randomly sample 20% of the images from train for each sequence.
        # first train images
        
        # Now process images in self.path_images_orig. 
        path_train = jp(self.path_data, 'train')
        path_test = jp(self.path_data, 'test')
        # create folders train, val, test in self.path_images_orig
        dirs.create_folders_within_folder(self.path_images_orig, ['train', 'val', 'test'])
        dirs.create_folders_within_folder(jp(self.path_images_orig, 'train'), ['gt'])
        dirs.create_folders_within_folder(jp(self.path_images_orig, 'val'), ['gt'])
        dirs.create_folders_within_folder(jp(self.path_images_orig, 'test'), ['gt'])
        # create folders in gt
        dirs.create_folders_within_folder(jp(self.path_images_orig, 'train', 'gt'), ['BinarySegmentation', 'TypeSegmentation', 'TypeSegmentationRescaled'])
        dirs.create_folders_within_folder(jp(self.path_images_orig, 'val', 'gt'), ['BinarySegmentation', 'TypeSegmentation', 'TypeSegmentationRescaled'])
        dirs.create_folders_within_folder(jp(self.path_images_orig, 'test', 'gt'), ['BinarySegmentation', 'TypeSegmentation', 'TypeSegmentationRescaled'])
        # list train sequences
        train_sequences = dirs.list_folders(path_train)
        # do the following only if folders are empty
        if len(os.listdir(jp(self.path_images_orig, 'val', 'gt'))) == 0 or self.config['overwrite_img']:
            # for each sequence, randomly sample 20% of the images for val
            for seq in tqdm(train_sequences, desc='Cropping and copying train and val images'):
                # list images in left_frames
                left_frame_images = dirs.list_files(jp(path_train, seq, 'left_frames'))
                # randomly sample 20% of the images for val
                val_images = np.random.choice(left_frame_images, size=int(len(left_frame_images)*0.2), replace=False)
                train_images = list(set(left_frame_images) - set(val_images))
                # copy corresponding images in gt
                for val_img in tqdm(val_images, desc='val ' + seq, leave=False, colour='yellow'):
                    # read val_img
                    vali = io.read_image(jp(path_train, seq, 'left_frames', val_img))
                    # crop vali
                    vali = vali[28:vali.shape[0]-28, 320:vali.shape[1]-320, :] # as suggested in https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/Data/
                    assert vali.shape[0] == 1024 and vali.shape[1] == 1280, 'Image not cropped correctly'
                    # save vali in self.path_images_orig/val
                    io.save_image(vali, jp(self.path_images_orig, 'val', 'seq' + seq.split('_')[-1] +val_img))
                    # read corresponding gt images
                    gt_bin = io.read_image(jp(path_train, seq, 'gt', 'BinarySegmentation', val_img))
                    gt_bin = gt_bin[28:gt_bin.shape[0]-28, 320:gt_bin.shape[1]-320]
                    gt_type = io.read_image(jp(path_train, seq, 'gt', 'TypeSegmentation', val_img))
                    gt_type = gt_type[28:gt_type.shape[0]-28, 320:gt_type.shape[1]-320]
                    gt_type_rescaled = io.read_image(jp(path_train, seq, 'gt', 'TypeSegmentationRescaled', val_img))
                    gt_type_rescaled = gt_type_rescaled[28:gt_type_rescaled.shape[0]-28, 320:gt_type_rescaled.shape[1]-320]
                    # save gt images in self.path_images_orig/val/gt
                    io.save_image(gt_bin, jp(self.path_images_orig, 'val', 'gt', 'BinarySegmentation', 'seq' + seq.split('_')[-1] +val_img))
                    io.save_image(gt_type, jp(self.path_images_orig, 'val', 'gt', 'TypeSegmentation', 'seq' + seq.split('_')[-1] +val_img))
                    io.save_image(gt_type_rescaled, jp(self.path_images_orig, 'val', 'gt', 'TypeSegmentationRescaled', 'seq' + seq.split('_')[-1] +val_img))
                for train_img in tqdm(train_images, desc='train ' + seq, leave=False, colour='yellow'):
                    # read train_img
                    traini = io.read_image(jp(path_train, seq, 'left_frames', train_img))
                    # crop traini
                    traini = traini[28:traini.shape[0]-28, 320:traini.shape[1]-320, :]
                    assert traini.shape[0] == 1024 and traini.shape[1] == 1280, 'Image not cropped correctly'
                    # save traini in self.path_images_orig/train
                    io.save_image(traini, jp(self.path_images_orig, 'train', 'seq' + seq.split('_')[-1] + train_img))
                    # read corresponding gt images
                    gt_bin = io.read_image(jp(path_train, seq, 'gt', 'BinarySegmentation', train_img))
                    gt_bin = gt_bin[28:gt_bin.shape[0]-28, 320:gt_bin.shape[1]-320]
                    gt_type = io.read_image(jp(path_train, seq, 'gt', 'TypeSegmentation', train_img))
                    gt_type = gt_type[28:gt_type.shape[0]-28, 320:gt_type.shape[1]-320]
                    gt_type_rescaled = io.read_image(jp(path_train, seq, 'gt', 'TypeSegmentationRescaled', train_img))
                    gt_type_rescaled = gt_type_rescaled[28:gt_type_rescaled.shape[0]-28, 320:gt_type_rescaled.shape[1]-320]
                    # save gt images in self.path_images_orig/train/gt
                    io.save_image(gt_bin, jp(self.path_images_orig, 'train', 'gt', 'BinarySegmentation', 'seq' + seq.split('_')[-1] +train_img))
                    io.save_image(gt_type, jp(self.path_images_orig, 'train', 'gt', 'TypeSegmentation', 'seq' + seq.split('_')[-1] +train_img))
                    io.save_image(gt_type_rescaled, jp(self.path_images_orig, 'train', 'gt', 'TypeSegmentationRescaled', 'seq' + seq.split('_')[-1] +train_img))
        # copy test images
        test_sequences = dirs.list_folders(path_test)
        for seq in tqdm(test_sequences, desc='Cropping and copying test images'):
            test_images = dirs.list_files(jp(path_test, seq, 'left_frames'))
            for test_img in tqdm(test_images, desc='test ' + seq, leave=False, colour='yellow'):
                # if image already exists (assuming masks exist too), skip
                if os.path.exists(jp(self.path_images_orig, 'test', 'seq' + seq.split('_')[-1] + test_img)) and not self.config['overwrite_img']:
                    print('Skipping', test_img, 'seq', seq)
                    continue
                # read test_img
                testi = io.read_image(jp(path_test, seq, 'left_frames', test_img))
                # crop testi
                testi = testi[28:testi.shape[0]-28, 320:testi.shape[1]-320, :]
                assert testi.shape[0] == 1024 and testi.shape[1] == 1280, 'Image not cropped correctly'
                # save testi in self.path_images_orig/test
                io.save_image(testi, jp(self.path_images_orig, 'test', 'seq' + seq.split('_')[-1] + test_img))
                # read corresponding gt images
                gt_bin = io.read_image(jp(path_test, seq, 'gt', 'BinarySegmentation', test_img))
                gt_bin = gt_bin[28:gt_bin.shape[0]-28, 320:gt_bin.shape[1]-320]
                gt_type = io.read_image(jp(path_test, seq, 'gt', 'TypeSegmentation', test_img))
                gt_type = gt_type[28:gt_type.shape[0]-28, 320:gt_type.shape[1]-320]
                gt_type_rescaled = io.read_image(jp(path_test, seq, 'gt', 'TypeSegmentationRescaled', test_img))
                gt_type_rescaled = gt_type_rescaled[28:gt_type_rescaled.shape[0]-28, 320:gt_type_rescaled.shape[1]-320]
                # save gt images in self.path_images_orig/test/gt
                io.save_image(gt_bin, jp(self.path_images_orig, 'test', 'gt', 'BinarySegmentation', 'seq' + seq.split('_')[-1] +test_img))
                io.save_image(gt_type, jp(self.path_images_orig, 'test', 'gt', 'TypeSegmentation', 'seq' + seq.split('_')[-1] +test_img))
                io.save_image(gt_type_rescaled, jp(self.path_images_orig, 'test', 'gt', 'TypeSegmentationRescaled', 'seq' + seq.split('_')[-1] +test_img))
        # Now resize images (only, not GT) and save them in self.path_images_output
        # create train, val an test folders within path_images_output
        dirs.create_folders_within_folder(self.path_images_output, ['train', 'val', 'test'])
        for subset in ['train', 'val', 'test']:
            # list images
            images = dirs.list_files(jp(self.path_images_orig, subset))
            for image in tqdm(images, desc='Resizing images ' + subset, colour='blue'):
                if not os.path.exists(jp(self.path_images_output, subset, image)) or self.config['overwrite_img']:
                    image_processing.resize_and_save(jp(self.path_images_orig, subset, image), jp(self.path_images_output, subset, image), size = self.config['size'])
                else:
                    print("Skipping resizing of image", image, "File exists and overwrite_img is set to False")
        
    # create data
    def create_dataset(self):
        printer.print_section('Creating QA pairs')
        # Now generate questions about regions and about whole images for each split
        mapping_index2tool_no_other = copy.deepcopy(self.mapping_index2tool)
        del mapping_index2tool_no_other[7] # exclude Other class
        for subset in ['train', 'val', 'test']:
            if os.path.exists(jp(self.path_qa_output, subset + 'qa.json')) and not self.config['overwrite_qa']:
                print("Skipping qa file", subset + '_qa.json', "File exists and overwrite_qa is set to False")
                continue # if qa file exists and no overwrite is required, skip

            qa = [] # to save QA entries

            # list all images in current subset (original size)
            images = [e for e in dirs.list_files(jp(self.path_images_orig, subset))]
            # for each image, generate a set of questions about random regions, based on the labels that are present in the image (balanced)
            print(subset, 'set...')
            for i_image, image in enumerate(tqdm((images), desc='Creating QA pairs for ' + subset, colour='blue')):
                # get mask
                mask = image.replace('endo', 'mask')
                # read image
                img = io.read_image(jp(self.path_images_orig, subset, image))
                # read mask
                mask = io.read_image(jp(self.path_images_orig, subset, 'gt', 'TypeSegmentation', mask)) # working with TypeSegmentation

                # get unique labels in mask
                labels = [e for e in list(set(mask.flatten())) if e in set(self.mapping_index2tool.keys()) and e != 7] # exlude Other

                # for each label, generate binary mask and then generate pairs of questions about regions
                for l in labels:
                    # create binary mask
                    mask_bin = np.zeros_like(mask)
                    mask_bin[mask == l] = 1
                    # generate pairs of questions
                    partial_qa_id = '1' +  str(l).zfill(2) + str(i_image+1).zfill(4)
                    qa_pairs = qa_factory.generate_questions_about_regions(self.config, mask_bin, self.mapping_index2tool[l], partial_qa_id, image, balanced=True, dataset='sts2017')
                    qa.extend(qa_pairs)

                # questions about any tool in the image
                l = 99 # special label for any tool
                mask_bin = np.zeros_like(mask)
                mask_bin[mask > 0] = 1
                # generate pairs
                partial_qa_id = '1' +  str(l).zfill(2) + str(i_image+1).zfill(4)
                qa_pairs = qa_factory.generate_questions_about_regions(self.config, mask_bin, 'any tool or instrument', partial_qa_id, image, balanced=True, dataset='sts2017')

                # questions about whole image - is there [tool] in this region?
                #qa_pairs = qa_factory.generate_questions_about_image(self.config, labels, mapping_index2tool_no_other, image, img.shape[0], img.shape[1], str(i_image+1).zfill(4))
                #qa.extend(qa_pairs)

            io.save_json(qa, jp(self.path_qa_output, subset + '_qa.json'))

    # process dataset
    def process_dataset(self):
        pass
        self.built_success = True



class DME(object):
    # Class to convert DME data to format of RIS-VQA and INSEGCAT-VQA datasets
    def __init__(self, config):
        self.path_dme_orig = config['path_orig']
        self.path_qa = jp(self.path_dme_orig, 'qa')
        self.path_images = jp(self.path_dme_orig, 'visual')
        self.path_masks = jp(self.path_dme_orig, 'masks')
        self.path_dme_output = config['path_output']
        os.makedirs(self.path_dme_output, exist_ok=True)
        dirs.create_folders_within_folder(self.path_dme_output, ['images', 'qa'])
        self.subsets = ['train', 'val', 'test']
        self.dict_question_types = {'whole': 'whole', 'inside':'region', 'grade': 'grade', 'fovea': 'macula'}

        self.objects = {'hard exudates': 'he', 'optic discs': 'od', 'grade': 'grade'}

        self.success = False

    def print_details(self):
        print('DME dataset')
        print('Original data path:', self.path_dme_orig)
        print('Output data path:', self.path_dme_output)

    def build(self):
        # here is where the conversion occurs
        # for each subset, read the json file
        for subset in self.subsets:
            os.makedirs(jp(self.path_dme_output, 'images', subset), exist_ok=True)
            qa_group = []
            # read json file
            qa = io.read_json(jp(self.path_qa, subset + 'qa.json'))
            # now for each qa pair, create a new entry with the following structure:
            for entry in tqdm(qa, desc='Processing ' + subset + ' set', colour='blue'):
                # open mask
                mask = io.read_image(jp(self.path_masks, subset, 'maskA', entry['mask_name']))
                props = regionprops(mask)
                assert len(props) == 1, 'There should be only one region in the mask'
                # get bounding box
                miny, minx, maxy, maxx = props[0].bbox
                if miny == 0 and minx == 0 and maxy == mask.shape[0] and maxx == mask.shape[1]:
                    # whole image
                    question_alt = entry['question']
                    mask_coords = ((0, 0), mask.shape[0], mask.shape[1])
                else:
                    assert entry['question_type'] == 'inside', 'Question type should be inside'
                    question_alt = entry['question'].replace('in this region', 'in the ellipse contained in the bounding box ({}, {}, {}, {})'.format(miny, minx, maxy, maxx))
                    mask_coords = ((miny, minx), maxy-miny, maxx-minx)
                qa_group.append({
                                    'image_name': entry['image_name'],
                                    'question': entry['question'],
                                    'question_alt': question_alt,
                                    'question_id': entry['question_id'],
                                    'question_type': self.dict_question_types[entry['question_type']],
                                    'mask_coords': mask_coords, #(top_left_resized, window_h_resized, window_w_resized),
                                    'mask_coords_orig': None,
                                    'answer': entry['answer'],
                                    'mask_size': (mask.shape[0], mask.shape[1]),
                                    'mask_size_orig': None, # not necessary for DME
                                    'question_object': self.dict_question_types[entry['question_type']]
                })
                # copy image using shutil if it wasn't copied before
                if not os.path.exists(jp(self.path_dme_output, 'images', subset, entry['image_name'])):
                    shutil.copy(jp(self.path_images, subset, entry['image_name']), jp(self.path_dme_output, 'images', subset, entry['image_name']))
            # save qa_group
            io.save_json(qa_group, jp(self.path_dme_output, 'qa', subset + '_qa.json'))
        self.success = True

    def created_successfully(self):
        if self.success:
            print('Dataset was created successfully.')
        else:
            print('Dataset was not created successfully.')