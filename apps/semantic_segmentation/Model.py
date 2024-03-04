#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:24:54 2023

@author: peter
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 25 10:14:35 2023

@author: fusco_p
"""
#%%


import torch
import semantic_segmentation.Configuration
import torchvision
import segmentation_models_pytorch as smp


#%%
# USEFUL TO SHORTLIST SPECIFIC CLASSES IN DATASETS WITH LARGE NUMBER OF CLASSES
select_classes = ['background', 'road']


ENCODER         = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
CLASSES         = select_classes
ACTIVATION      = 'sigmoid'         # could be None for logits or 'softmax2d' for multiclass segmentation

# CREATE SEGMENTATION MODEL WITH PRETRAINED ENCODER
UNet_Model = smp.Unet(
                      encoder_name    = ENCODER, 
                      encoder_weights = ENCODER_WEIGHTS, 
                      classes         = len(CLASSES), 
                      activation      = ACTIVATION
                    )

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


#%%
