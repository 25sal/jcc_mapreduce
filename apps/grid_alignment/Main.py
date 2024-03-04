#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 16:40:22 2023

@author: peter
"""

# %%

import numpy             as np
import random
import matplotlib.pyplot as plt
import Configuration     as Config
import pathlib
import pandas            as pd


from Utility_Functions import Data_Generator
from System_Manager    import System_Manager_Grid


# %%


if __name__ == '__main__':

    # **********************************
    # BOX SIZE
    # **********************************
    ENDS_X             = [0, 10]
    ENDS_Y             = [0, 10]
    NX                 = 3
    NY                 = 3
    TH                 = 10
    PITCHX             = 1
    PITCHY             = 1
    GRID_DELTA_X       = 0
    GRID_DELTA_Y       = 0
    GENERATED_DATA     = False
    GENERATED_AS_LINES = True
    

    # ***************
    # GENERATED DATA
    # ***************
    if GENERATED_DATA:

        
        if GENERATED_AS_LINES:
            
            vertical_lines_ends, horizontal_lines_ends = Data_Generator().get_data_lines( X_BOUNDS               = [0, 1000],
                                                                                          Y_BOUNDS               = [0, 500],
                                                                                          DIV_X                  = 5,
                                                                                          DIV_Y                  = 2,
                                                                                          LEN_BOUNDS             = [20, 25],
                                                                                          VERTICAL_ORIENTATION   = [0, 15],
                                                                                          HORIZONTAL_ORIENTATION = [0, 15],
                                                                                          SEED                   = 42,
                                                                                          ADD_NOISE              = False )
            
            
            input_data   = np.vstack( [vertical_lines_ends.reshape(-1, 2), horizontal_lines_ends.reshape(-1, 2) ] )
            x_input_data = input_data[:, 0]
            y_input_data = input_data[:, 1]
            
        else:
            
            
            x_input_data, y_input_data = Data_Generator.get_data_points(ENDS_X,
                                                                        ENDS_Y,
                                                                        NX,
                                                                        NY,
                                                                        PITCHX,
                                                                        PITCHY,
                                                                        DELTA     = [GRID_DELTA_X, GRID_DELTA_Y],
                                                                        ADD_NOISE = False )    


    else:

        # ***************
        # IMPORTED DATA
        # ***************
        CWD                  = pathlib.Path.cwd()
        IMPUT_DATA_PATH      = CWD.parent / 'Mask_Prediction'
        SOURCE_NAME          = 'Capodrise'
        SOURCE_DATA_FILENAME = IMPUT_DATA_PATH / SOURCE_NAME / 'MASK_WITH_MARKERS' / f'{SOURCE_NAME}_Markers_Data.csv'

        input_data   = pd.read_csv(SOURCE_DATA_FILENAME, header=None)
        x_input_data = input_data.iloc[:, 0::2].to_numpy().reshape(-1, 1).squeeze( axis = 1 )
        y_input_data = input_data.iloc[:, 1::2].to_numpy().reshape(-1, 1).squeeze( axis = 1 )


# %%

    Manager_Grid = System_Manager_Grid( x_input_data,
                                        y_input_data,
                                        GRID_WIDTH    = [800,  1800],
                                        GRID_HEIGHT   = [1250, 1750],
                                        NX            = 4,
                                        NY            = 1,
                                        TH_PERCENTAGE = Config.TH_PERCENTAGE,
                                        config        = Config )


# %%

    # ********************
    # PLOTTING INPUT DATA
    # ********************
    # Manager_Grid.plot_initial_data( AS_LINES      = True,
    #                                 COLOR         = 'blue',
    #                                 INCLUDE_LABEL = True )


# %%

    # *********************
    # PLOTTING INITIAL GRID
    # *********************
    # Manager_Grid.plot_initial_grid( MAIN_LINE  = True,
    #                                 STRIP_LINE = True )


# %%

    # *****************************
    # ROTATING AND TRANSLATING DATA
    # *****************************
    # Manager_Grid.modify_data( delta_x       = 50,
    #                           delta_y       = 100,
    #                           rot_angle_deg = 45,
    #                           rotation_axis = [0.0, 0.0],
    #                           sequence      = 'RT' )


# %%

    # ********************************************
    # PLOTTING ROTATED AND TRANSLATED INITIAL DATA
    # ********************************************
    # Manager_Grid.plot_modified_data( AS_LINES      = True,
    #                                   COLOR         = 'blue',
    #                                   INCLUDE_LABEL = True )


# %%

    # *****************************
    # ROTATING AND TRANSLATING GRID
    # *****************************
    # Manager_Grid.modify_grid( delta_x       = -50,
    #                           delta_y       = -100,
    #                           rot_angle_deg = -45,
    #                           rotation_axis = Manager_Grid.rotation_axis_data,
    #                           sequence      = 'TR' )


# %%

    # ********************************************
    # PLOTTING ROTATED AND TRANSLATED INITIAL GRID
    # ********************************************
    # Manager_Grid.plot_modified_grid( MAIN_LINE  = True,
    #                                   STRIP_LINE = True )


# %%

    # *******************************************************
    # COUNTING THE NUMBER OF LINES INSIDER THE GRID'S STRIPS
    # *******************************************************
    # Manager_Grid.count_lines_inside_grid()


# %%

    # ****************************************
    # RUNNING GENETIC ALGORITHM
    # ****************************************
    # Manager_Grid.run( ACTIVITIES = ['data', 'grid'],
    #                   DELAY      = False )

    Manager_Grid.run()


# %%

    # ****************************************
    # PLOTTING BEST INDIVIDUAL AVAILABLE
    # ****************************************
    # Manager_Grid.plot_best_individual()


# %%
