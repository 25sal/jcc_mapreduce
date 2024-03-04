#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 10:58:21 2023

@author: peter
"""

#%%

import pathlib
import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import cv2


from System_Manager_IP import System_Manager


#%%

if __name__ == '__main__':
    
    CWD_PATH        = pathlib.Path.cwd()
    INPUT_MASK_PATH = CWD_PATH / 'Input_Data' / 'Mask.png'
    
    
    #********************************
    # INSTANTIATE A SYSTEM MAN OBJECT
    #********************************
    manager = System_Manager()
    
    
    #************************************
    # GET LINES LYING ON INPUT MASK EDGES
    #************************************
    lines, centroids = manager.get_objs_on_mask_image( mask_path                          = str( INPUT_MASK_PATH ),
                                                       DISTANCE_RESOLUTION_IN_PIXELS      = 1, 
                                                       ANGLE_RESOLUTION_IN_RADIANS        = np.pi/180, 
                                                       MIN_NUMBER_OF_VOTES_FOR_VALID_LINE = 100, 
                                                       MIN_ALLOWED_LENGTH_OF_LINE         = 10, 
                                                       MAX_GAP_BETWEEN_LINE_FOR_JOINING   = 250,
                                                       X_TOLERANCE                        = 0,
                                                       Y_TOLERANCE                        = 0,
                                                       MAX_LINE_LENGTH                    = 0 )


#%%
    image = cv2.imread( str( INPUT_MASK_PATH ) )
    
    cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
    
    
    for line in lines:
        
        image = cv2.line( image, (line[0], line[1]), (line[2], line[3]),  (0, 255, 0), 2 ) 
          
    
    cv2.imshow('Display', image) 
    
    
#%%
