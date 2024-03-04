#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:24:22 2023

@author: peter
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 29 11:35:39 2023

@author: fusco_p
"""

#%%


import torch
import pathlib
import torchvision
import Configuration                             as Config
import segmentation_models_pytorch               as smp
import segmentation_models_pytorch.utils.metrics
import pandas                                    as pd
import numpy                                     as np
import copy
import matplotlib.pyplot                         as plt


from sklearn.model_selection import train_test_split
from System_Manager          import System_Manager
from semantic_segmentation.Dataset                 import SegmentationDataset
from Model                   import UNet_Model
from Model                   import preprocessing_fn
from Utility_Functions       import get_training_augmentation
from Utility_Functions       import get_preprocessing
from Utility_Functions       import one_hot_encode
from Utility_Functions       import visualize
from Utility_Functions       import colour_code_segmentation
from Utility_Functions       import reverse_one_hot
from Utility_Functions       import get_validation_augmentation


#%%

if __name__ == '__main__':
    
    
    CWD = pathlib.Path.cwd()
    
    class_dict = pd.read_csv( Config.DATASET_PATH.parent / "label_class_dict.csv")
    
    
    #****************
    # GET CLASS NAMES
    #****************
    class_names = class_dict['name'].tolist()
    
    
    #*********************
    # GET CLASS RGB VALUES
    #*********************
    class_rgb_values = class_dict[ ['r', 'g', 'b'] ].values.tolist()
    
    
    #******************************************************************************
    # USEFUL TO SHORTLIST SPECIFIC CLASSES IN DATASETS WITH LARGE NUMBER OF CLASSES
    #******************************************************************************
    select_classes = ['background', 'road']
    
    
    #***********************************
    # GET RGB VALUES OF REQUIRED CLASSES
    #***********************************
    select_class_indices    = [ class_names.index(cls.lower()) for cls in select_classes ]
    select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]
    
    
    #%%
    
    if Config.IMAGE_SUFFIX == 'png':
        
        
        #*****************************************************
        # LOAD THE IMAGE AND MASK FILEPATHS IN A SORTED MANNER
        #*****************************************************
        imagePaths = sorted( list( pathlib.Path( Config.IMAGE_DATASET_PATH ).glob( '*.png' ) ) )
        maskPaths  = sorted( list( pathlib.Path( Config.MASK_DATASET_PATH  ).glob( '*.png' ) ) )  
    
    
    
        #*****************************************************************
        # PARTITION THE DATA INTO TRAINING, VALIDATION AND TESTING SPLITS
        #*****************************************************************
        X_train, X_test, y_train, y_test = train_test_split( imagePaths, 
                                                             maskPaths,
                                                             test_size    = Config.TEST_SPLIT, 
                                                             random_state = 42 )
    
    
        X_valid, X_test, y_valid, y_test = train_test_split( X_test, 
                                                             y_test,
                                                             test_size    = 0.5, 
                                                             random_state = 42 )
        
        
    elif Config.IMAGE_SUFFIX == 'tiff':
        
        
        X_train = list( ( pathlib.Path( Config.IMAGE_DATASET_PATH.parent ) / 'train' ).glob( f'*.{Config.IMAGE_SUFFIX}' ) )
        y_train = list( ( pathlib.Path( Config.IMAGE_DATASET_PATH.parent ) / 'train_labels' ).glob( f'*.{Config.IMAGE_SUFFIX}' ) )
        
        X_valid = list( ( pathlib.Path( Config.IMAGE_DATASET_PATH.parent ) / 'val' ).glob( f'*.{Config.IMAGE_SUFFIX}' ) )
        y_valid = list( ( pathlib.Path( Config.IMAGE_DATASET_PATH.parent ) / 'val_labels' ).glob( f'*.{Config.IMAGE_SUFFIX}' ) )

        X_test  = list( ( pathlib.Path( Config.IMAGE_DATASET_PATH.parent ) / 'test' ).glob( f'*.{Config.IMAGE_SUFFIX}' ) )
        y_test  = list( ( pathlib.Path( Config.IMAGE_DATASET_PATH.parent ) / 'test_labels' ).glob( f'*.{Config.IMAGE_SUFFIX}' ) )
        

#%%


    #***********************************
    # CREATE DATASETS
    #***********************************
    trainDataSet = SegmentationDataset( images_paths     = X_train, 
                                        masks_paths      = y_train,
                                        augmentation     = get_training_augmentation(),
                                        preprocessing    = get_preprocessing(preprocessing_fn),
                                        class_rgb_values = select_class_rgb_values,
                                        one_hot_encode   = one_hot_encode )


    validDataSet = SegmentationDataset( images_paths     = X_valid, 
                                        masks_paths      = y_valid,
                                        augmentation     = get_validation_augmentation(),
                                        preprocessing    = get_preprocessing(preprocessing_fn),
                                        class_rgb_values = select_class_rgb_values,
                                        one_hot_encode   = one_hot_encode )
    
    
    testDataSet = SegmentationDataset( images_paths     = X_test, 
                                       masks_paths      = y_test,
                                       augmentation     = get_validation_augmentation(),
                                       preprocessing    = get_preprocessing(preprocessing_fn),
                                       class_rgb_values = select_class_rgb_values,
                                       one_hot_encode   = one_hot_encode )
    

#%%


    #***********************************
    # CREATE DATALOADERS
    #***********************************
    trainDataLoader = torch.utils.data.DataLoader( trainDataSet, 
                                                   shuffle     = True,
                                                   batch_size  = Config.TRAIN_BATCH_SIZE,
                                                   pin_memory  = Config.PIN_MEMORY,
                                                   num_workers = 0 )


    validDataLoader = torch.utils.data.DataLoader( validDataSet, 
                                                   shuffle     = True,
                                                   batch_size  = Config.VALID_BATCH_SIZE,
                                                   pin_memory  = Config.PIN_MEMORY,
                                                   num_workers = 0 )
    
    
    testDataLoader = torch.utils.data.DataLoader( testDataSet, 
                                                  shuffle     = False,
                                                  batch_size  = Config.TEST_BATCH_SIZE,
                                                  pin_memory  = Config.PIN_MEMORY,
                                                  num_workers = 0 )


#%%

    #**************************
    # INITIALIZE MODEL
    #**************************
    unet_model = UNet_Model
    

#%%

    #**************************
    # DEFINE LOSS FUNCTION
    #**************************
    criterion = smp.utils.losses.DiceLoss()


#%%

    #**************************
    # DEFINE OPTIMIZER
    #**************************
    optimizer = torch.optim.Adam( unet_model.parameters(), 
                                  lr = Config.LEARNING_RATE )


#%%

    #***************
    # DEFINE METRICS
    #***************
    metrics = [ smp.utils.metrics.IoU(threshold = 0.5) ]
    
    
#%%


    #***************************
    # INSTANTIATE SYSTEM MANAGER
    #***************************
    Manager = System_Manager( model           = unet_model,
                              trainDataLoader = trainDataLoader,
                              validDataLoader = validDataLoader,
                              testDataLoader  = testDataLoader,
                              criterion       = criterion,
                              optimizer       = optimizer,
                              metrics         = metrics,
                              config          = Config )
    

#%%
    
    #*************************************
    #PERFORM TRAINING AND VALIDATION STEPS
    #*************************************
    Manager.train()
    
    
#%%

    #********************
    #PERFORM TESTING STEP
    #********************
    Manager.test( select_class_rgb_values,
                  LOAD_MODEL = False )
    
    
#%%

    #***********************
    #PERFORM PREDICTION STEP
    #***********************
    # image, mask = testDataSet[0]

    # predicted_mask = Manager.predict( image, 
    #                                   select_class_rgb_values,
    #                                   LOAD_MODEL = True )
    
    
    
    # image = np.transpose( image, (2, 1, 0))
    # mask  = np.transpose( mask,  (2, 1, 0))
    # visualize( original_image       = image,
    #             ground_truth_mask    = colour_code_segmentation( reverse_one_hot(mask), select_class_rgb_values ),
    #             one_hot_encoded_mask = reverse_one_hot(mask) )


    # visualize( original_image       = image,
    #             ground_truth_mask    = colour_code_segmentation( reverse_one_hot(predicted_mask), select_class_rgb_values ),
    #             one_hot_encoded_mask = reverse_one_hot(predicted_mask) )


#%%
