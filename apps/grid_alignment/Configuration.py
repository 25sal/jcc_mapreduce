# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 20:26:36 2023

@author: fusco_p
"""
#%%


import numpy as np


#%%


#*************************************
# GENETIC ALGORITHM PARAMETERS SECTION
#*************************************
POPULATION_SIZE   = 200
MAX_GENERATIONS   = 30
HALL_OF_FAME_SIZE = 1
P_CROSSOVER       = 0.8                 #===> PROBABILITY FOR CROSSOVER
P_MUTATION        = 0.1                 #===> PROBABILITY FOR MUTATING AN INDIVIDUAL
INDEPENDENT_PROB  = 0.02
CROWDING_FACTOR   = 20.0                #===> CROWDING FACTOR FOR CROSSOVER AND MUTATION
NUM_GENES         = 3


#*************************************
# GRID PARAMETERS SECTION
#*************************************
CENTER_X_BOUNDS       = [0, 2000] 
CENTER_Y_BOUNDS       = [0, 2000]
ROTATION_BOUNDS       = [0, 45]
GRID_WIDTH            = [0, 1500]
GRID_HEIGHT           = [0, 1500]
NX                    = 3
NY                    = 3
TH_PERCENTAGE         = 0.12             #===> THAT MEANS #% OF THE MINIMUM OF THE HALF OF THE SPACE
TH                    = 0                #     BETWEEN TWO SUBSEQUNT MAIN VERTICAL OR HORIZONTAL LINES 
GRID_DELTA_X          = 0
GRID_DELTA_Y          = 0
ROT_ANGLE_DEG         = 0
ROTATION_AXIS         = [0.0, 0.0]
MU                    = 0
STD                   = 0.05
MAIN_LINE_GRID_COLOR  = 'red'
STRIP_LINE_GRID_COLOR = 'green'
OPENCV_COORD_SYS      = True
PLOTTING_PAD          = 0.3             #===> THAT MEANS #% OF THE MAXIMUM DISTANCE


#*************************************
# RANSAC ALGORITHM PARAMETERS SECTION
#*************************************
NUMBER_OF_ITERATIONS  = np.inf
TOLERANCE             = 3
MAX_INLIERS_COUNT     = 0
PROB_OUTLIER          = 0.5
DESIRED_PROB          = 0.95
SAMPLES_PERCENTAGE    = 0.2             #===> THAT MEANS 20% OF ALL INPUT DATA


#%%