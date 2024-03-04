#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 10:59:37 2023

@author: peter
"""

#%%

import cv2
import numpy as np


#%%

class System_Manager():
    
    
    @staticmethod
    def get_objs_on_mask_image( mask_path,
                                DISTANCE_RESOLUTION_IN_PIXELS      = 1,
                                ANGLE_RESOLUTION_IN_RADIANS        = np.pi/20,
                                MIN_NUMBER_OF_VOTES_FOR_VALID_LINE = 10,
                                MIN_ALLOWED_LENGTH_OF_LINE         = 30,
                                MAX_GAP_BETWEEN_LINE_FOR_JOINING   = 30,
                                X_TOLERANCE                        = 5,
                                Y_TOLERANCE                        = 5,
                                MAX_LINE_LENGTH                    = 0 ):
        
        '''
        #Parameters:
            #    mask_path (string)...: The path of the input mask
            

        #Returns:
            #    lines     (numpy array [lines x (x1, y1, x2, y2)])...: The array containing the coordinates of the lines
            #    centroids (numpy array [centroids x (x, y)]).........: The array containing the coordinates of the centroids
        '''
        
        #*****************************************
        # CONTANER FOR LINES AND CENTROIDS TO SAVE        
        #*****************************************
        lines_to_save     = []
        centroids_to_save = []
        

        #*************************************
        # LOAD THE IMAGE USING MASK IMAGE PATH
        #*************************************
        mask_image = cv2.imread( mask_path )
        
        
        #**************************************
        # CONVERT INPUT MASK IMAGE TO GREYSCALE
        #**************************************
        mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        
            
        #************************************************************************
        #FINDING HORIZONTAL AND VERTICAL LINES USING PROBABILISTIC HOUGH TRANFORM
        #************************************************************************
        lines = cv2.HoughLinesP( image         = mask_image_gray,                              # --> INPUT IMAGE
                                 rho           = DISTANCE_RESOLUTION_IN_PIXELS, 
                                 theta         = ANGLE_RESOLUTION_IN_RADIANS, 
                                 threshold     = MIN_NUMBER_OF_VOTES_FOR_VALID_LINE, 
                                 minLineLength = MIN_ALLOWED_LENGTH_OF_LINE, 
                                 maxLineGap    = MAX_GAP_BETWEEN_LINE_FOR_JOINING )
        
        
        #********************************
        # CALCULATE CENTROIDS COORDINATES
        #********************************
        for line in lines:
            
            for x1, y1, x2, y2 in line:
                

                #****************************************
                # CALCULATE THE LENGTH OF THE CURRET LINE
                #****************************************
                line_length = np.sqrt( (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) )
                
                
                #****************************************************************
                # ALLOWING HORIZONTAL AND VERTICAL LINES IN A SPECIFIC RANGE ONLY
                #****************************************************************
                if ( ( ( np.abs(x2 - x1) < X_TOLERANCE ) or ( np.abs(y2 - y1) < Y_TOLERANCE )) and 
                     ( line_length > MAX_LINE_LENGTH )                                ):
                    
                    X = np.mean( [x1, x2] ).astype('int')
                    Y = np.mean( [y1, y2] ).astype('int')
                    
                    
                    #*********************************
                    # SAVING VALID CENTROIDS AND LINES
                    #*********************************
                    centroids_to_save.append( [X, Y] )
                    lines_to_save.append( [x1, y1, x2, y2] )

    
        return np.array( lines_to_save ), np.array( centroids_to_save )
    

#%%
