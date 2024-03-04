#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:53:15 2023

@author: peter
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 17:20:47 2023

@author: peter
"""

#%%
import os
import torch
import cv2
import numpy                       as np
import albumentations              as album
import segmentation_models_pytorch as smp
import pandas                      as pd
import matplotlib.pyplot           as plt


#%%


def to_tensor( x, 
               **kwargs ):
    
    return x.transpose(2, 0, 1).astype('float32')


#********************************
# DEFINING TRAINING AUGMENTATIONS
#********************************
def get_training_augmentation():
    
    train_transform = [    
                        album.RandomCrop( height       = 256, 
                                          width        = 256, 
                                          always_apply = True ),
                        album.OneOf( [
                                        album.HorizontalFlip(p = 1),
                                        album.VerticalFlip(p = 1),
                                        album.RandomRotate90(p = 1),
                                      ],
                                      p = 0.75, ),
                       ]
    
    return album.Compose( train_transform )


#**********************************
# DEFINING VALIDATION AUGMENTATIONS
#**********************************
def get_validation_augmentation():
    
    #**********************************************************
    # ADD SUFFICIENT PADDING TO ENSURE IMAGE IS DIVISIBLE BY 32
    #**********************************************************
    test_transform = [ album.PadIfNeeded( min_height   = 1536, 
                                          min_width    = 1536, 
                                          always_apply = True, 
                                          border_mode  = 0 ), ]
    
    return album.Compose( test_transform )


#********************************
#GETTING PIPELINE TRANSFORMATIONS
#********************************
def get_preprocessing( preprocessing_fn = None ):
    """Construct preprocessing transform    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """   
    _transform = []
    if preprocessing_fn:
    
        _transform.append(album.Lambda(image=preprocessing_fn))
        
    _transform.append( album.Lambda( image = to_tensor, 
                                     mask  = to_tensor ) )
        
    return album.Compose(_transform)


#***************************************************
# PERFORM ONE HOT ENCODING ON LABEL
#***************************************************
def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        
        equality  = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        
        semantic_map.append(class_map)
        
    semantic_map = np.stack(semantic_map, axis = -1)

    return semantic_map


#***************************************************
# PERFORM REVERSE ONE-HOT-ENCODING ON LABELS / PREDS
#***************************************************
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    x = np.argmax(image, axis = -1)
    return x


#*****************************************************
# PERFORM COLOUR CODING ON THE REVERSE-ONE-HOT OUTPUTS
#*****************************************************
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x            = colour_codes[image.astype(int)]

    return x



#*************************************************************************************************
# CENTER CROP PADDED IMAGE / MASK TO ORIGINAL IMAGE DIMS
#*************************************************************************************************
def crop_image(image, target_image_dims = [1500, 1500, 3]):
   
    target_size = target_image_dims[0]
    image_size  = len(image)
    padding     = (image_size - target_size) // 2

    if padding < 0:
        
        return image

    return image[ padding:image_size - padding, padding:image_size - padding, :, ]


#***************************************
# HELPER FUNCTION FOR DATA VISUALIZATION
#***************************************
def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure( figsize = (20, 8) )
    for idx, (name, image) in enumerate(images.items()):
        
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]); 
        plt.yticks([])
        
        # get title from the parameter names
        plt.title( name.replace('_', ' ').title(), fontsize = 20)
        plt.imshow(image)
        
    plt.show()
    
    
#%%