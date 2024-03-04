#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:25:23 2023

@author: peter
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 25 08:38:42 2023

@author: fusco_p
"""

#%%

#******************************
# IMPORT THE NECESSARY PACKAGES
#******************************


import torch
import pathlib
import os
import datetime
import collections


#%%

#******************************
# CURRENT SESSION DATE AND TIME
#******************************
DATE_TIME_STRING = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")


BASE_PATH = pathlib.Path('/home/spark/apps')

#*************************
# BASE PATH OF THE DATASET
#*************************
IMAGE_SUFFIX = 'tiff'
DATASET_PATH = BASE_PATH / pathlib.Path( "Massachusetts_Roads_Dataset" )
DATASET_PATH = DATASET_PATH / f'{IMAGE_SUFFIX}'


#************************************************
# DEFINE THE PATH TO THE IMAGES AND MASKS DATASET
#************************************************
IMAGE_DATASET_PATH = DATASET_PATH / "Images"
MASK_DATASET_PATH  = DATASET_PATH / "Masks"


#**********************
# DEFINE THE TEST SPLIT
#**********************
TEST_SPLIT  = 0.2


#************************************************************
# DETERMINE THE DEVICE TO BE USED FOR TRAINING AND EVALUATION
#************************************************************
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


#***********************************************************
# DETERMINE IF WE WILL BE PINNING MEMORY DURING DATA LOADING
#***********************************************************
PIN_MEMORY = True if DEVICE == "cuda" else False


#***************************************************************
# DEFINE THE NUMBER OF CHANNELS IN THE INPUT, NUMBER OF CLASSES,
# AND NUMBER OF LEVELS IN THE U-NET MODEL
#***************************************************************
NUM_CHANNELS = 1
NUM_CLASSES  = 1
NUM_LEVELS   = 3


#*****************************************************************
# INITIALIZE LEARNING RATE, NUMBER OF EPOCHS TO TRAIN FOR, AND THE
# BATCH SIZE
#*****************************************************************
LEARNING_RATE    = 0.00008
NUM_EPOCHS       = 40
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
TEST_BATCH_SIZE  = 1




#*********************************************
# DEFINE THE PATH TO THE MODEL TO LOAD
#*********************************************
MODEL_TO_LOAD = BASE_PATH / pathlib.Path( 'Models/BEST_MODEL_03_11_2023_17_36_20.pth' )


#**********************************
# DEFINE THE INPUT IMAGE DIMENSIONS
#**********************************
INPUT_IMAGE_WIDTH  = 1500
INPUT_IMAGE_HEIGHT = 1500


#**********************************************
# DEFINE IF THE TRAINED MODEL NEEDS TO BE SAVED
#**********************************************
SAVE_TRAINED_MODEL = True


#********************************************
# DEFINE THRESHOLD TO FILTER WEAK PREDICTIONS
#********************************************
THRESHOLD = 0.5


#*********************************************
# DEFINE THE PATH TO THE BASE OUTPUT DIRECTORY
#*********************************************
BASE_PATH_DICT = { 'TRAIN' : {},
                   'TEST'  : {} }

BASE_OUTPUT = pathlib.Path.cwd()
BASE_OUTPUT = pathlib.Path('/home/spark/apps')

#************************************************
# DEFINE THE PATH TO THE SESSION OUTPUT DIRECTORY
#************************************************
RUN_PATH = BASE_OUTPUT / '_'.join( ["RUN_PATH", DATE_TIME_STRING] )


#***************************************************************
# DEFINE THE PATH TO THE OUTPUT SERIALIZED MODEL
#***************************************************************
BASE_PATH_DICT['TRAIN']['MODEL_PATH'] = RUN_PATH / "BEST_MODEL"
BASE_PATH_DICT['TRAIN']['MODEL_NAME'] = 'Best_Model.pth'


#***********************************************************************
# DEFINE THE PATH TO SAVE PREDICTION WHEN TRAING IS PREVIOUSLY PERFORMED
#***********************************************************************
BASE_PATH_DICT['TEST']['PREDICTED_PATH'] = RUN_PATH / 'TEST/PREDICTED_MASK'
BASE_PATH_DICT['TEST']['ORIGINAL_PATH']  = RUN_PATH / 'TEST/ORIGINAL_MASK'


#***************************************************************
# DEFINE THE PATH TO SAVE ACCURACY AND LOSS PLOTS 
#***************************************************************
BASE_PATH_DICT['TRAIN']['OUTPUT_PLOT_PATH']  = RUN_PATH / 'OUTPUT_PLOT'


#***************************************************************
# DEFINE THE PATH TO SAVE OUTPUT STUFF
#***************************************************************
BASE_PATH_DICT['TRAIN']['OUTPUT_PARAMS_FILE_PATH']  = RUN_PATH / 'OUTPUT_FILES'
BASE_PATH_DICT['TRAIN']['OUTPUT_PARAMS_FILENAME']   = BASE_PATH_DICT['TRAIN']['OUTPUT_PARAMS_FILE_PATH'] / 'Parameters.dat'



#*************************************
# LIST OF ALL PATHS USED IN THE SYSTEM
#*************************************
ALL_PATHS = collections.defaultdict( list )
for keys, paths in BASE_PATH_DICT.items():
    
    for key, path in paths.items():
        
        if key.find('PATH') != -1:
            
            ALL_PATHS[keys].append( path )
        

#****************************************************
# DICTIONARY OF ALL PARAMETERS DEFINED IN THIS MODULE
#****************************************************
ALL_PARAMS = { name : str( value ) for name, value in vars().items() if not name.startswith("__") and name.isupper() }


#%%
