#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 16:44:34 2023

@author: peter
"""

#%%

import numpy             as np
import matplotlib.pyplot as plt


#%%

class Data_Generator():
        
    
    @classmethod
    def get_data_points( cls,
                         ENDS_X,
                         ENDS_Y,
                         NX,
                         NY,
                         PITCHX,
                         PITCHY,
                         ROT_ANGLE_DEG = 0,
                         ROTATION_AXIS = [0.0, 0.0],
                         DELTA         = [0.0, 0.0],
                         ADD_NOISE     = False ):    
        
        
        mask = np.zeros((PITCHY*NY + (NY + 1), PITCHX*NX + (NX + 1)), dtype = int)
        
        rows = list( range(0, mask.shape[0], PITCHY + 1) )
        cols = list( range(0, mask.shape[1], PITCHX + 1) )
        
        mask[rows, :] = 1
        mask[:, cols] = 1
    
        X = np.linspace( *ENDS_X, mask.shape[1])
        Y = np.linspace( *ENDS_Y, mask.shape[0])
    
        X_noise = np.zeros_like(X)
        Y_noise = np.zeros_like(Y)
    
        if ADD_NOISE:
            
            MU       = 0.0
            STD      = 0.05 * np.std(X) # for %5 Gaussian noise
            X_noise += np.random.normal( MU, STD, len(X))
            Y_noise += np.random.normal( MU, STD, len(Y))
        
    
        #************
        #ADDING NOISE
        #************
        X = X + X_noise
        Y = Y + Y_noise
    
        X_Grid, Y_Grid = np.meshgrid(X, Y)
        
        
        x_input_data = X_Grid.flatten()[ mask.flatten().astype(bool) ]
        y_input_data = Y_Grid.flatten()[ mask.flatten().astype(bool) ] 
        
        rot_angle_rad = np.deg2rad( ROT_ANGLE_DEG )
    
        rotateMatrix = np.array( [ [ np.cos( -rot_angle_rad ), -np.sin( -rot_angle_rad ) ],
                                   [ np.sin( -rot_angle_rad ),  np.cos( -rot_angle_rad ) ] ] )
    
        
        stacked_xy_data = np.column_stack( [ x_input_data.flatten(order = 'F'), 
                                             y_input_data.flatten(order = 'F') ] )
        rotated_xy_grid = ( stacked_xy_data - ROTATION_AXIS) @ rotateMatrix + ROTATION_AXIS
        
        return ( rotated_xy_grid[:, 0].reshape( x_input_data.shape, order = 'F' ) + DELTA[0], 
                 rotated_xy_grid[:, 1].reshape( y_input_data.shape, order = 'F' ) + DELTA[1] )


        
    
    @classmethod
    def get_data_lines( cls,
                        X_BOUNDS,
                        Y_BOUNDS,
                        DIV_X,
                        DIV_Y,
                        LEN_BOUNDS,
                        VERTICAL_ORIENTATION,
                        HORIZONTAL_ORIENTATION,
                        SEED,
                        ADD_NOISE = False ):


        if SEED:
            
            np.random.seed(SEED)
        
        # noise = np.random.normal(mu, std, size = x.shape)
        
        vertical_lines_x_centroid   = np.linspace( X_BOUNDS[0], X_BOUNDS[1], DIV_X )
        vertical_lines_y_centroid   = np.linspace( Y_BOUNDS[0], Y_BOUNDS[1], DIV_Y )
        
        horizontal_lines_x_centroid = np.linspace( X_BOUNDS[0], X_BOUNDS[1], DIV_X )
        horizontal_lines_y_centroid = np.linspace( Y_BOUNDS[0], Y_BOUNDS[1], DIV_Y )

            
        vertical_lines_x_centroid, vertical_lines_y_centroid = np.meshgrid( vertical_lines_x_centroid,
                                                                            vertical_lines_y_centroid )
        
        vertical_lines_x_centroid = vertical_lines_x_centroid.flatten() 
        vertical_lines_y_centroid = vertical_lines_y_centroid.flatten()

                
        horizontal_lines_x_centroid, horizontal_lines_y_centroid = np.meshgrid( horizontal_lines_x_centroid,
                                                                                horizontal_lines_y_centroid )
        
        horizontal_lines_x_centroid = horizontal_lines_x_centroid.flatten() 
        horizontal_lines_y_centroid = horizontal_lines_y_centroid.flatten() 

        
        vertical_lines_lenght            = np.random.uniform( LEN_BOUNDS[0], LEN_BOUNDS[1], size = vertical_lines_x_centroid.shape )    
        horizontal_lines_lenght          = np.random.uniform( LEN_BOUNDS[0], LEN_BOUNDS[1], size = horizontal_lines_x_centroid.shape )    
        
        lines_vertical_orient   = np.random.uniform( np.deg2rad( VERTICAL_ORIENTATION[0] ),   np.deg2rad( VERTICAL_ORIENTATION[1] ),   size = vertical_lines_x_centroid.shape )
        lines_horizontal_orient = np.random.uniform( np.deg2rad( HORIZONTAL_ORIENTATION[0] ), np.deg2rad( HORIZONTAL_ORIENTATION[1] ), size = horizontal_lines_x_centroid.shape )
        
        vertical_lines_ends = np.column_stack( [ vertical_lines_x_centroid - 0.5*vertical_lines_lenght*np.sin( lines_vertical_orient ), 
                                                 vertical_lines_y_centroid + 0.5*vertical_lines_lenght*np.cos( lines_vertical_orient ), 
                                                 vertical_lines_x_centroid + 0.5*vertical_lines_lenght*np.sin( lines_vertical_orient ), 
                                                 vertical_lines_y_centroid - 0.5*vertical_lines_lenght*np.cos( lines_vertical_orient ) ] )

        horizontal_lines_ends = np.column_stack( [ horizontal_lines_x_centroid - 0.5*horizontal_lines_lenght*np.cos( lines_horizontal_orient ), 
                                                   horizontal_lines_y_centroid + 0.5*horizontal_lines_lenght*np.sin( lines_horizontal_orient ), 
                                                   horizontal_lines_x_centroid + 0.5*horizontal_lines_lenght*np.cos( lines_horizontal_orient ), 
                                                   horizontal_lines_y_centroid - 0.5*horizontal_lines_lenght*np.sin( lines_horizontal_orient ) ] )
        
    
        return vertical_lines_ends, horizontal_lines_ends 
    
    
#%%


def get_initial_data( ENDS_X,
                      ENDS_Y,
                      NX,
                      NY,
                      PITCHX,
                      PITCHY,
                      ROT_ANGLE_DEG = 0,
                      ROTATION_AXIS = [0.0, 0.0],
                      DELTA         = [0.0, 0.0],
                      ADD_NOISE     = False ):    
    
    
    mask = np.zeros((PITCHY*NY + (NY + 1), PITCHX*NX + (NX + 1)), dtype = int)
    
    rows = list( range(0, mask.shape[0], PITCHY + 1) )
    cols = list( range(0, mask.shape[1], PITCHX + 1) )
    
    mask[rows, :] = 1
    mask[:, cols] = 1

    X = np.linspace( *ENDS_X, mask.shape[1])
    Y = np.linspace( *ENDS_Y, mask.shape[0])

    X_noise = np.zeros_like(X)
    Y_noise = np.zeros_like(Y)

    if ADD_NOISE:
        
        MU       = 0.0
        STD      = 0.05 * np.std(X) # for %5 Gaussian noise
        X_noise += np.random.normal( MU, STD, len(X))
        Y_noise += np.random.normal( MU, STD, len(Y))
    

    #************
    #ADDING NOISE
    #************
    X = X + X_noise
    Y = Y + Y_noise

    X_Grid, Y_Grid = np.meshgrid(X, Y)
    
    
    x_input_data = X_Grid.flatten()[ mask.flatten().astype(bool) ]
    y_input_data = Y_Grid.flatten()[ mask.flatten().astype(bool) ] 
    
    rot_angle_rad = np.deg2rad( ROT_ANGLE_DEG )

    rotateMatrix = np.array( [ [ np.cos( -rot_angle_rad ), -np.sin( -rot_angle_rad ) ],
                               [ np.sin( -rot_angle_rad ),  np.cos( -rot_angle_rad ) ] ] )

    
    stacked_xy_data = np.column_stack( [ x_input_data.flatten(order = 'F'), 
                                         y_input_data.flatten(order = 'F') ] )
    rotated_xy_grid = ( stacked_xy_data - ROTATION_AXIS) @ rotateMatrix + ROTATION_AXIS
    
    return ( rotated_xy_grid[:, 0].reshape( x_input_data.shape, order = 'F' ) + DELTA[0], 
             rotated_xy_grid[:, 1].reshape( y_input_data.shape, order = 'F' ) + DELTA[1] )


#%%