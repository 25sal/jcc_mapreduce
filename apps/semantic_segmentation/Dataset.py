#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:26:28 2023

@author: peter
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 25 10:09:45 2023

@author: fusco_p
"""
#%%

import torch
import cv2
import os

from torch.utils.data import Dataset


#%%


class SegmentationDataset( torch.utils.data.Dataset ):

    """Massachusetts Roads Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
                    self, 
                    images_paths, 
                    masks_paths, 
                    class_rgb_values = None, 
                    augmentation     = None, 
                    preprocessing    = None,
                    one_hot_encode   = None
                ):
        

        self.images_paths = images_paths
        self.masks_paths  = masks_paths

        self.class_rgb_values = class_rgb_values
        self.augmentation     = augmentation
        self.preprocessing    = preprocessing
        self.one_hot_encode   = one_hot_encode
        
    
    def __getitem__(self, i):
        
        
        #************************
        # READ IMAGES AND MASKS
        #************************
        image = cv2.cvtColor( cv2.imread( str( self.images_paths[i] ) ), cv2.COLOR_BGR2RGB )
        mask  = cv2.cvtColor( cv2.imread( str( self.masks_paths[i] ) ),  cv2.COLOR_BGR2RGB )
        

        #************************
        # ONE-HOT-ENCODE THE MASK
        #************************
        mask = self.one_hot_encode( mask, 
                                    self.class_rgb_values ).astype('float')
        
        
        #************************
        # APPLY AUGMENTATIONS
        #************************
        if self.augmentation:
        
            sample = self.augmentation( image = image, 
                                        mask  = mask )
            image  = sample['image']
            mask   = sample['mask']
        
        
        #************************
        # APPLY PREPROCESSING
        #************************
        if self.preprocessing:
        
            sample = self.preprocessing( image = image, 
                                         mask  = mask )
            image  = sample['image']
            mask   = sample['mask']
            
        return image, mask
        
    def __len__(self):
    
        #*****************
        # RETURN LENGTH OF 
        #*****************
        return len(self.images_paths)

        
        
#%%