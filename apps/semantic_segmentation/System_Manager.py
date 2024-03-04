#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:52:48 2023

@author: peter
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 25 16:36:15 2023

@author: fusco_p
"""

#%%

# import torchvision 
import semantic_segmentation.Configuration
import torch
import os
import time
import matplotlib.pyplot           as plt
import pathlib
import shutil
import collections
import numpy                       as np
import cv2
import segmentation_models_pytorch as smp
import copy
import pandas                      as pd


from semantic_segmentation.Model                   import UNet_Model
from semantic_segmentation.Dataset                 import SegmentationDataset
from sklearn.model_selection import train_test_split
from imutils                 import paths
from tqdm                    import tqdm
from semantic_segmentation.Utility_Functions       import colour_code_segmentation, crop_image, reverse_one_hot


#%%


class System_Manager():
    
    def __init__( self,
                  model,
                  trainDataLoader,
                  validDataLoader,
                  testDataLoader,
                  criterion,
                  optimizer,
                  metrics,
                  config ):
        
        self.model           = model
        self.trainDataLoader = trainDataLoader
        self.validDataLoader = validDataLoader
        self.testDataLoader  = testDataLoader
        self.criterion       = criterion
        self.optimizer       = optimizer
        self.metrics         = metrics
        self.config          = config
        self.train_logs_list = []
        self.valid_logs_list = []
        self.best_model      = None

        
        self.train_epoch = smp.utils.train.TrainEpoch(
                                                       model, 
                                                       loss      = self.criterion, 
                                                       metrics   = self.metrics, 
                                                       optimizer = self.optimizer,
                                                       device    = self.config.DEVICE,
                                                       verbose   = True,
                                                      )
            
        self.valid_epoch = smp.utils.train.ValidEpoch(
                                                       model, 
                                                       loss    = self.criterion, 
                                                       metrics = self.metrics, 
                                                       device  = self.config.DEVICE,
                                                       verbose = True,
                                                      )
        
        


    def setting_environment( self,
                             MODE ):
        
        #*****************************************
        # CHECKING IF ALL FOLDER HAVE BEEN CREATED
        #*****************************************
        for path in self.config.ALL_PATHS[MODE]:
            
            path.mkdir(parents = True)
            
        
    def train( self ):


        #********************
        # SETTING ENVIRONMENT
        #********************
        self.setting_environment( MODE = 'TRAIN' )
        
        
        self.model.train()
        self.model.to( self.config.DEVICE)
        
        best_iou_score = 0.0


        for epoch in range( self.config.NUM_EPOCHS ):
    
            #******************************
            # PERFORM TRAINING & VALIDATION
            #******************************
            print( '\nEpoch: {0}/{1}'.format(epoch + 1, self.config.NUM_EPOCHS) )
            
            train_logs = self.train_epoch.run( self.trainDataLoader )
            valid_logs = self.valid_epoch.run( self.validDataLoader )
            
            self.train_logs_list.append(train_logs)
            self.valid_logs_list.append(valid_logs)
            
    
            #*************************************************
            # SAVE MODEL IF A BETTER VAL IOU SCORE IS OBTAINED
            #*************************************************
            if best_iou_score < valid_logs['iou_score']:
                
                best_iou_score = valid_logs['iou_score']
                self.best_model = copy.deepcopy( self.model )
                

        #****************************************************
        #SAVING THE MODEL AFTER TRAINING AND VALIDATION STEPS
        #****************************************************
        if self.config.SAVE_TRAINED_MODEL:
            
            self.save_model()


        #****************************************************
        #SAVING ALL THE PARAMETERS OF THE MODEL
        #****************************************************
        self.save_params()
        
        
        #****************************
        #SAVING PLOTS OF METRICS USED
        #****************************
        self.getMetricsPlots()
                
    
    def test( self,
              select_class_rgb_values,
              LOAD_MODEL = False ):
        
        
        #********************
        # SETTING ENVIRONMENT
        #********************
        self.setting_environment( MODE = 'TEST' )


        if LOAD_MODEL:
            
            print("Loading model")
            self.load_model( self.config.MODEL_TO_LOAD )
        
        
        print("Performing Testing")
        
        for idx, (image, mask) in enumerate( self.testDataLoader, 0 ):

            
            image = image.to( self.config.DEVICE )
            
            
            #***********************************
            # PREDICT TEST IMAGE
            #***********************************
            predicted_mask = self.model(image)
            predicted_mask = predicted_mask.detach().squeeze().cpu().numpy()
            
            
            #****************************************************
            # CONVERT PRED_MASK FROM 'CHW' FORMAT TO 'HWC' FORMAT
            #****************************************************
            predicted_mask = np.transpose( predicted_mask, (1, 2, 0) )
            predicted_mask = crop_image( colour_code_segmentation( reverse_one_hot(predicted_mask), 
                                                                   select_class_rgb_values) )
        
            
            #*******************************************************************
            # SAVING PREDICTED MASK USING PATH DEFINED IN THE CONFIGURATION FILE
            #*******************************************************************
            cv2.imwrite( str( self.config.BASE_PATH_DICT['TEST']['PREDICTED_PATH'] / 'Predicted_mask_{0}.png'.format(idx) ), predicted_mask )
                
            
            mask = mask.detach().squeeze().cpu().numpy()
            # mask = np.transpose( mask, (2, 1, 0))
            mask = np.transpose( mask, (1, 2, 0) )
            mask = crop_image( colour_code_segmentation( reverse_one_hot(mask), 
                                                         select_class_rgb_values) )
            

            #*******************************************************************
            # SAVING ORIGINAL MASK USING PATH DEFINED IN THE CONFIGURATION FILE
            #*******************************************************************
            cv2.imwrite( str( self.config.BASE_PATH_DICT['TEST']['ORIGINAL_PATH'] / 'Original_mask_{0}.png'.format(idx) ), mask )
                
                
    def predict( self,
                 image,
                 select_class_rgb_values,
                 LOAD_MODEL = False ):
    
        
        if LOAD_MODEL:
            
            print("Loading model")
            self.load_model( self.config.MODEL_TO_LOAD )
        
        
        print("Performing Prediction")
        
        #***********************************
        # CHECKING IF IMAGE IS A NUMPY ARRAY
        #***********************************
        if isinstance( image, (np.ndarray) ):
            
            image = torch.tensor( image, device = self.config.DEVICE )
        
        image = image.to( self.config.DEVICE ).unsqueeze(dim = 0)
        
        #***********************************
        # PREDICT TEST IMAGE
        #***********************************
        predicted_mask = self.model(image)
        predicted_mask = predicted_mask.detach().squeeze().cpu().numpy()
        
        
        #****************************************************
        # CONVERT PRED_MASK FROM 'CHW' FORMAT TO 'HWC' FORMAT
        #****************************************************
        # predicted_mask = np.transpose( predicted_mask, (1, 2, 0) )
        # predicted_mask = crop_image( colour_code_segmentation( reverse_one_hot(predicted_mask), 
        #                                                         select_class_rgb_values) )
        
        
        return predicted_mask
        
        
    def save_model( self ):

        
        torch.save( self.best_model, ( self.config.BASE_PATH_DICT['TRAIN']['MODEL_PATH'] / '_'.join( [self.config.BASE_PATH_DICT['TRAIN']['MODEL_PATH'].name, self.config.DATE_TIME_STRING ] ) ).with_suffix( '.pth' ) ) 


    def save_params( self ):
        
        
        MAX_LEN = np.max( list( map(len, self.config.ALL_PARAMS.keys() ) ) )
        
        with open( self.config.BASE_PATH_DICT['TRAIN']['OUTPUT_PARAMS_FILENAME'], 'w' ) as fid:
            
            for key, value in self.config.ALL_PARAMS.items():
                
                output_string = f'{key:{MAX_LEN}s} : {value}\n'
                
                fid.write( output_string )
            
        
    def load_model( self,
                    model_to_load_filename ):
        
        
        self.model = None
        self.model = torch.load( model_to_load_filename, map_location = torch.device('cpu') ).to( self.config.DEVICE )
        
        
    def getMetricsPlots( self ):
            
            
        #**********************************************
        # PLOT DICE LOSS & IOU METRIC FOR TRAIN VS. VAL
        #**********************************************
        self.train_logs_df = pd.DataFrame( self.train_logs_list )
        self.valid_logs_df = pd.DataFrame( self.valid_logs_list )
        self.train_logs_df.T
    
        fig, ax = plt.subplots(figsize = (20, 8))
        
        ax.plot( [ item + 1 for item in self.train_logs_df.index.tolist() ], self.train_logs_df.iou_score.tolist(), lw = 3, label = 'Train')
        ax.plot( [ item + 1 for item in self.valid_logs_df.index.tolist() ], self.valid_logs_df.iou_score.tolist(), lw = 3, label = 'Valid')
        
        ax.set_xlabel('Epochs',        fontsize = 21)
        ax.set_ylabel('IoU Score',     fontsize = 21)
        ax.set_title('IoU Score Plot', fontsize = 21)
        ax.legend(loc ='best',         fontsize = 16)
        ax.grid()
        
        fig.savefig( self.config.BASE_PATH_DICT['TRAIN']['OUTPUT_PLOT_PATH'] / 'iou_score_plot.png')
        
        

        fig, ax = plt.subplots(figsize = (20, 8))
        
        plt.plot( [ item + 1 for item in self.train_logs_df.index.tolist() ], self.train_logs_df.dice_loss.tolist(), lw = 3, label = 'Train')
        plt.plot( [ item + 1 for item in self.valid_logs_df.index.tolist() ], self.valid_logs_df.dice_loss.tolist(), lw = 3, label = 'Valid')
        
        plt.xlabel('Epochs',        fontsize = 21)
        plt.ylabel('Dice Loss',     fontsize = 21)
        plt.title('Dice Loss Plot', fontsize = 21)
        plt.legend(loc ='best',     fontsize = 16)
        plt.grid()
        
        fig.savefig( self.config.BASE_PATH_DICT['TRAIN']['OUTPUT_PLOT_PATH'] / 'dice_loss_plot.png')


        
#%%
