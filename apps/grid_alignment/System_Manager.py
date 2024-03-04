#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 17:16:39 2023

@author: peter
"""

#%%

import random
import matplotlib.animation as animation
import numpy                as np
import matplotlib.pyplot    as plt
import time


from deap import base
from deap import creator
from deap import tools
from deap import algorithms


#%%

class System_Manager_Grid():
    
    
    def __init__( self,
                  x_input_data,
                  y_input_data,
                  GRID_WIDTH,
                  GRID_HEIGHT,
                  NX,
                  NY,
                  TH_PERCENTAGE,
                  config ):
        
        
        self.x_input_data                       = x_input_data[:]      
        self.y_input_data                       = y_input_data[:]
        self.modified_x_input_data              = x_input_data[:]
        self.modified_y_input_data              = y_input_data[:]
        self.x_data_strip_grid_lines            = np.array(0)
        self.x_data_strip_grid_lines            = np.array(0)
        self.y_data_strip_grid_lines            = np.array(0)
        self.y_data_strip_grid_lines            = np.array(0) 
        self.modified_x_global_main_grid_lines  = np.array(0)
        self.modified_y_global_main_grid_lines  = np.array(0)
        self.modified_x_global_strip_grid_lines = np.array(0)
        self.modified_y_global_strip_grid_lines = np.array(0)
        self.bounding_box                       = np.array( [ [ np.min(self.x_input_data), np.max(self.x_input_data) ], 
                                                              [ np.min(self.y_input_data), np.max(self.y_input_data) ] ] )
        self.num_lines_inside_grid              = 0
        self.idx_lines_inside_grid              = []
        self.x_input_data_rot_center            = None
        self.y_input_data_rot_center            = None
        self.config                             = config
        self.rotation_axis_data                 = [ 0.0, 0.0 ] 
        self.rotation_axis_grid                 = [ 0.0, 0.0 ] 
                
        
        #**************************
        # SYSTEM MANAGER FLAGS
        #**************************
        self.grid_is_built = False
        
        
        #**************************
        # GENETIC ALGORITHM SECTION
        #**************************
        self.toolbox = base.Toolbox()
        
        
        #******************************************************************************************
        # CREATING AN OPERATOR THAT RETURNS A LIST OF ATTRIBUTES IN THE DESIRED RANGE AND DIMENSION
        #******************************************************************************************
        self.toolbox.register("attributesCreator", self.in_range_creator )

        
        #************************************************************************************************
        # DEFINING A SINGLE OBJECTIVE, MAXIMIZING FITNESS STRATEGY
        #************************************************************************************************
        creator.create("FitnessMax", base.Fitness, weights = (+1.0,))
        
        
        #************************************************************************************************
        # CREATING THE INDIVIDUAL CLASS BASED ON LIST OF INTEGERS
        #************************************************************************************************
        creator.create("Individual", list, fitness = creator.FitnessMax)


        #**************************************************************************************************
        # CREATING THE INDIVIDUAL CREATION OPERATOR TO FILL UP AN INDIVIDUAL INSTANCE WITH SHUFFLED INDICES
        #**************************************************************************************************
        self.toolbox.register("individualCreator", tools.initIterate, creator.Individual, self.toolbox.attributesCreator)
        
        
        #************************************************************************************************
        # CREATING THE POPULATION CREATION OPERATOR TO GENERATE A LIST OF INDIVIDUALS
        #************************************************************************************************
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individualCreator)
        
        
        #************************************************************************************************
        # TOURNAMENT SELECTION WITH TOURNAMENT SIZE OF 3
        #************************************************************************************************
        self.toolbox.register("select", tools.selTournament, tournsize = 3 )
        
        
        #************************************************************************************************
        # CXSIMULATEDBINARYBOUNDED CROSSOVER
        #************************************************************************************************
        self.toolbox.register( "mate", 
                                tools.cxSimulatedBinaryBounded, 
                                low = [ self.config.CENTER_X_BOUNDS[0], self.config.CENTER_Y_BOUNDS[0], self.config.ROTATION_BOUNDS[0] ], 
                                up  = [ self.config.CENTER_X_BOUNDS[1], self.config.CENTER_Y_BOUNDS[1], self.config.ROTATION_BOUNDS[1] ], 
                                eta = self.config.CROWDING_FACTOR)
        
        
        #************************************************************************************************
        # MUTPOLYNOMIALBOUNDED MUTATION
        # INDPB: INDEPENDENT PROBABILITY FOR EACH ATTRIBUTE TO BE FLIPPED
        #************************************************************************************************
        self.toolbox.register( "mutate", 
                                tools.mutPolynomialBounded, 
                                low   = [ self.config.CENTER_X_BOUNDS[0], self.config.CENTER_Y_BOUNDS[0], self.config.ROTATION_BOUNDS[0] ], 
                                up    = [ self.config.CENTER_X_BOUNDS[1], self.config.CENTER_Y_BOUNDS[1], self.config.ROTATION_BOUNDS[1] ], 
                                eta   = self.config.CROWDING_FACTOR, 
                                indpb = 1.0/self.config.NUM_GENES )
        
        
        #************************************************************************************************
        # REGISTERING THE COST FUNCTION
        #************************************************************************************************
        self.toolbox.register("evaluate", self.get_residual)


        #************************************************************************************************
        # INITIALIZING STATISTICS ACCUMULATORS
        #************************************************************************************************
        self.maxFitnessValues        = []
        self.meanFitnessValues       = []
        self.best_individual         = None
        self.best_individual_history = []


        #************************************************************************************************
        # CREATING GRID
        #************************************************************************************************
        self.create_grid( GRID_WIDTH    = GRID_WIDTH,
                          GRID_HEIGHT   = GRID_HEIGHT,
                          NX            = NX,
                          NY            = NY,
                          TH_PERCENTAGE = TH_PERCENTAGE )


        #**********************************************************************
        # PLOTTING ELEMENTS SECTION
        #**********************************************************************
        plt.close( 'all' )
        
        self.fig = plt.figure( constrained_layout = True )
        spec_fig = self.fig.add_gridspec( ncols = 3, 
                                          nrows = 2 )
        
        
        #*********************
        # DEFINING AXES LIMITS
        #*********************
        self.x_limits = [0, 1]
        self.y_limits = [0, 1]        
        
        
        #*********************************************************
        # DEFINING DICTIONARY FOR STORING STATIC AND DYNAMIC PLOTS
        #*********************************************************
        self.axs = {}
        

        #*********************************************************
        # DEFINING STATIC PLOTS
        #*********************************************************
        self.axs['initial_static']    = self.fig.add_subplot( spec_fig[0, 0] )
        self.axs['data2grid_dynamic'] = self.fig.add_subplot( spec_fig[0, 1] )
        self.axs['grid2data_dynamic'] = self.fig.add_subplot( spec_fig[0, 2] )

        
        self.axis_title = [ 'Initial configuration (STATIC)',  'Data moves towards grid (DYNAMIC)', 'Grid moves towards data (DYNAMIC)' ]
        
        #************************
        # SETTING AXES PARAMETERS
        #************************
        for axis_title, (axis_name, axis_plot) in zip( self.axis_title, self.axs.items() ):
            
            axis_plot.set_title( axis_title )
            axis_plot.set_aspect( 'equal' )
            axis_plot.grid(True)
            
        
        #*********************************************************
        # UPDATING AXES LIMITS
        #*********************************************************
        self.update_axes_limits()
        

        #*********************************************************
        # SETTING AXES LIMITS
        #*********************************************************
        self.set_axes_limits()
        
        
        #*********************
        # DEFINING FINAL PLOTS
        #*********************
        self.fig_final = plt.figure( constrained_layout = True )
        spec_fig       = self.fig_final.add_gridspec( ncols = 2, 
                                                      nrows = 1 )
        
        #********************************************
        # DEFINING DICTIONARY FOR STORING FINAL PLOTS
        #********************************************
        self.axs_final = {}
        

        #*********************************************************
        # DEFINING FINAL PLOTS
        #*********************************************************
        self.axs_final['data2grid'] = self.fig_final.add_subplot( spec_fig[0, 0] )
        self.axs_final['grid2data'] = self.fig_final.add_subplot( spec_fig[0, 1] )
        
        self.axs_final['data2grid'].set_title( 'data2grid' )
        self.axs_final['data2grid'].set_aspect( 'equal' )
        self.axs_final['data2grid'].grid(True)

        self.axs_final['grid2data'].set_title( 'grid2data' )
        self.axs_final['grid2data'].set_aspect( 'equal' )
        self.axs_final['grid2data'].grid(True)
        
 
#%%

    def update_axes_limits( self ):
        
        #*********************************************************
        # FINDING AND SETTING AXES LIMITS
        #*********************************************************
        X_MAX = np.max( [ self.modified_x_input_data.max(), self.modified_x_global_strip_grid_lines.max(), self.x_data_strip_grid_lines.max() ] )
        X_MIN = np.min( [ self.modified_x_input_data.min(), self.modified_x_global_strip_grid_lines.min(), self.x_data_strip_grid_lines.min() ] ) 
        Y_MAX = np.max( [ self.modified_y_input_data.max(), self.modified_y_global_strip_grid_lines.max(), self.y_data_strip_grid_lines.max() ] )
        Y_MIN = np.min( [ self.modified_y_input_data.min(), self.modified_y_global_strip_grid_lines.min(), self.y_data_strip_grid_lines.min() ] )
        
        X_DIFF = X_MAX - X_MIN
        Y_DIFF = Y_MAX - Y_MIN
        
        
        X_MAX += np.abs( self.config.PLOTTING_PAD*X_DIFF )
        X_MIN -= np.abs( self.config.PLOTTING_PAD*X_DIFF )
        Y_MAX += np.abs( self.config.PLOTTING_PAD*Y_DIFF )
        Y_MIN -= np.abs( self.config.PLOTTING_PAD*Y_DIFF )
        
        
        self.x_limits = [X_MIN, X_MAX]
        self.y_limits = [Y_MIN, Y_MAX]    
        

#%%

    def set_axes_limits( self ):
        
        for axis in self.axs.values():
            
            axis.set_xlim( self.x_limits )
            axis.set_ylim( self.y_limits )
            
            
#%%

    def in_range_creator( self ):
        
        return [ random.uniform(low_bound, upper_bound) for low_bound, upper_bound in [ self.config.CENTER_X_BOUNDS, 
                                                                                        self.config.CENTER_Y_BOUNDS, 
                                                                                        self.config.ROTATION_BOUNDS ] ]


#%%

    def get_num_lines_inside_grid( self ):
        
        return self.num_lines_inside_grid


#%%

    def get_idx_lines_inside_grid( self ):
        
        return self.idx_lines_inside_grid[:]
    
    
#%%

    def get_residual( self,
                      individual ):
        
        
        DELTA_X, DELTA_Y, ROT_ANGLE_DEG = individual


        #***********************************************************************
        # UPDATING ORIGINAL X AND Y DATA APPLYING 
        # A DISPLACEMENT OF DELTA_X AND DELTA_Y PLUS A ROTATION OF ROT_ANGLE_DEG
        #***********************************************************************
        self.modify_data( delta_x       = DELTA_X, 
                          delta_y       = DELTA_Y, 
                          rot_angle_deg = ROT_ANGLE_DEG,
                          rotation_axis = [0.0, 0.0],
                          sequence      = 'RT' )
        
        
        #********************************************************************************
        # COUNTING THE NUMBER OF ROTATED AND TRASLATED LINES INSIDE ANY STRIP OF THE GRID
        #********************************************************************************
        self.count_lines_inside_grid()
        
        
        return self.get_num_lines_inside_grid(),
    

#%%

    def create_grid( self,
                     GRID_WIDTH    = None,
                     GRID_HEIGHT   = None,
                     NX            = None,
                     NY            = None,
                     TH_PERCENTAGE = None ):
        
        
        if GRID_WIDTH is not None:
            
            self.config.GRID_WIDTH = GRID_WIDTH
            
            
        if GRID_HEIGHT is not None:
    
            self.config.GRID_HEIGHT = GRID_HEIGHT
            

        if NX is not None:

            self.config.NX = NX
            
            
        if NY is not None:

            self.config.NY = NY
            
            
        if TH_PERCENTAGE is not None:
            
            self.config.TH_PERCENTAGE = TH_PERCENTAGE
        
        
        #*******************************************
        # GET COORDINATES FOR MAIN LINES OF THE GRID
        #*******************************************
        linesX = np.linspace( self.config.GRID_WIDTH[0],  self.config.GRID_WIDTH[1],  self.config.NX + 1)
        linesY = np.linspace( self.config.GRID_HEIGHT[0], self.config.GRID_HEIGHT[1], self.config.NY + 1)
        
        
        #*********************************************************************************************
        # TAKING THE PERCENTAGE OF THE MINIMUM SPACE BETWEEN THE VERTICAL OR THE HORIZONTAL MAIN LINES
        #*********************************************************************************************
        self.config.TH = self.config.TH_PERCENTAGE * np.min( [np.diff( linesX ).min(), np.diff( linesY ).min()] )
        
        
        #****************************************************
        # CREATING COORDINATES FOR THE MAIN LINES OF THE GRID
        #****************************************************
        self.vertical_main_lines_ends   = np.tile(linesX, 2).reshape(2, -1).T
        self.horizontal_main_lines_ends = np.tile(linesY, 2).reshape(2, -1).T
        
        
        #***********************************************************************
        # COPYING THE COORDINATES OF THE MAIN LINES FOR CREATING THE STRIP LINES
        #***********************************************************************
        self.vertical_strip_lines_ends   = self.vertical_main_lines_ends.copy()
        self.horizontal_strip_lines_ends = self.horizontal_main_lines_ends.copy()
        
        
        self.vertical_strip_lines_ends[:, 0] -= self.config.TH/2
        self.vertical_strip_lines_ends[:, 1] += self.config.TH/2

        self.horizontal_strip_lines_ends[:, 0] -= self.config.TH/2
        self.horizontal_strip_lines_ends[:, 1] += self.config.TH/2
        
        
        #************************************************************************
        # CREATING BOUNDING BOX MOST EXTERNAL FROM VERTICAL AND HORIZONTAL STRIPS
        #************************************************************************
        self.bounding_box = np.array( [ [self.vertical_strip_lines_ends[0][0],   self.vertical_strip_lines_ends[-1][-1]   ],
                                        [self.horizontal_strip_lines_ends[0][0], self.horizontal_strip_lines_ends[-1][-1] ] ] )
        
        
        #***************************************************************************************************
        # -------------------CREATING COORDINATE FOR THE MAIN LINES OF THE GRID - SECTION-------------------
        #***************************************************************************************************
        
        #*****************************************************************************************
        # REPEATING X COORDINATES BECAUSE A VERTICAL LINE HAS TWO X COORDINATES, ONE FOR EACH ENDS
        #*****************************************************************************************
        self.x_data_main_grid_lines = self.vertical_main_lines_ends
        
        
        #*************************************
        # UPDATING X COORDINATES OF GRID LINES
        #*************************************
        self.x_data_main_grid_lines  = np.concatenate( [ self.x_data_main_grid_lines, np.full( ( (self.config.NY + 1), 2), self.config.GRID_WIDTH ).astype('float') ], axis = 0) 

        
        #*************************************************************************************
        # REPEATING Y DATA BECAUSE AN HORIZONTAL LINE HAS TWO Y COORDINATES, ONE FOR EACH ENDS
        #*************************************************************************************
        self.y_data_main_grid_lines = np.full( ( (self.config.NX + 1), 2), self.config.GRID_HEIGHT ).astype('float')
        
        
        #*************************************
        # UPDATING Y COORDINATES OF GRID LINES
        #*************************************
        self.y_data_main_grid_lines = np.concatenate( [ self.y_data_main_grid_lines,  self.horizontal_main_lines_ends ], axis = 0)

        

        #****************************************************************************************************
        # -------------------CREATING COORDINATE FOR THE STRIP LINES OF THE GRID - SECTION-------------------
        #****************************************************************************************************
        
        
        #*****************************************************************************************
        # REPEATING X COORDINATES BECAUSE A VERTICAL LINE HAS TWO X COORDINATES, ONE FOR EACH ENDS
        #*****************************************************************************************
        self.x_data_strip_grid_lines = np.repeat( self.vertical_strip_lines_ends.flatten(), 2 ).astype(float).reshape(-1, 2)
        
        
        #*************************************
        # UPDATING X COORDINATES OF GRID LINES
        #*************************************
        self.x_data_strip_grid_lines  = np.concatenate( [ self.x_data_strip_grid_lines, 
                                                          np.full( (2*(self.config.NY + 1), 2), self.config.GRID_WIDTH ).astype('float') + (self.config.TH/2) * np.tile( [-1, 1], 2*(self.config.NY + 1)).reshape(-1, 2) ], axis = 0) 
        
        
        #*************************************************************************************
        # REPEATING Y DATA BECAUSE AN HORIZONTAL LINE HAS TWO Y COORDINATES, ONE FOR EACH ENDS
        #*************************************************************************************
        self.y_data_strip_grid_lines = np.full( (2*(self.config.NX + 1), 2), self.config.GRID_HEIGHT ).astype('float') + \
                                                (self.config.TH/2) * np.tile( [-1, 1], 2*(self.config.NX + 1)).reshape(-1, 2)
                
        
        # #*************************************
        # # UPDATING Y COORDINATES OF GRID LINES
        # #*************************************
        self.y_data_strip_grid_lines = np.concatenate( [ self.y_data_strip_grid_lines,
                                                         np.repeat( self.horizontal_strip_lines_ends.flatten(), 2 ).astype(float).reshape(-1, 2) ], axis = 0)

        
        self.grid_is_built = True


#%%

    def plot_initial_grid( self,
                           MAIN_LINE  = True,
                           STRIP_LINE = True,
                           PLOTS      = [ 'initial_static', 'data2grid_dynamic' ] ):
        
        #****************************************************
        #CHECK IF THE GRID WAS BUILT BEFORE TRYING TO PLOT IT
        #****************************************************
        if self.grid_is_built:
            
            
            if MAIN_LINE:
                
                for x_data, y_data in zip( self.x_data_main_grid_lines, 
                                           self.y_data_main_grid_lines ):
                
                    for which_plot in PLOTS:
                        
                        self.axs[which_plot].plot( x_data,
                                                   y_data,
                                                   color     = self.config.MAIN_LINE_GRID_COLOR,
                                                   linestyle = 'dashed',
                                                   linewidth = 1.5  )
                    
                
            if STRIP_LINE:
            
                for x_data, y_data in zip( self.x_data_strip_grid_lines, 
                                           self.y_data_strip_grid_lines ):
                
                    for which_plot in PLOTS:

                        self.axs[which_plot].plot( x_data,
                                                   y_data,
                                                   color     = self.config.STRIP_LINE_GRID_COLOR,
                                                   linestyle = 'dashed',
                                                   linewidth = 1.5  )    
                                    
        for which_plot in PLOTS:
            
            
            #******************************************************************
            # INVERTING Y AXIS IN ORDER TO KEEP COHERENCE WITH Y AXIS IN OPENCV
            #******************************************************************
            if self.config.OPENCV_COORD_SYS and ( not self.axs[which_plot].yaxis_inverted() ):
                
                self.axs[which_plot].invert_yaxis()
                
                
        #*********************************************************
        # UPDATING AXES LIMITS
        #*********************************************************
        self.update_axes_limits()
        

        #*********************************************************
        # SETTING AXES LIMITS
        #*********************************************************
        self.set_axes_limits()


#%%

    def plot_initial_data( self,
                           AS_LINES      = True,
                           COLOR         = 'blue',
                           INCLUDE_LABEL = True,
                           PLOTS         = [ 'initial_static', 'grid2data_dynamic' ] ):

        if AS_LINES:
            
            #*********************************
            # PLOTTING DATA USING SCATTER PLOT
            #*********************************
            for x_data, y_data in zip( self.x_input_data.reshape( (-1, 2) ), 
                                       self.y_input_data.reshape( (-1, 2) ) ):
                
                for which_plot in PLOTS:

                    self.axs[which_plot].plot( x_data, 
                                               y_data,
                                               color = COLOR )

            
        else:
            
            
            for which_plot in PLOTS:

                #*******************************************
                # PLOTTING DATA USING SCATTER ON STATIC PLOT
                #*******************************************
                self.axs[which_plot].scatter( self.x_input_data, 
                                              self.y_input_data,
                                              color = COLOR )
            
        
        #**************************************
        # INSERTING LABELS ON INPUT DATA POINTS
        #**************************************
        if INCLUDE_LABEL:
            
            if AS_LINES:
                
                #*********************************
                # PLOTTING DATA USING SCATTER PLOT
                #*********************************
                for idx, (x, y) in enumerate( zip( self.x_input_data.reshape( (-1, 2) ), 
                                                   self.y_input_data.reshape( (-1, 2) ) ), 1 ):
    
    
                    for which_plot in PLOTS:

                        self.axs[which_plot].text( x[0], 
                                                   y[0],
                                                   str(idx) )           
                    
                
            else:
                
                for idx, (x, y) in enumerate( zip( self.x_input_data, self.y_input_data ), 0 ):
                    
                    for which_plot in PLOTS:

                        #*****************************
                        # PLOTTING DATA ON STATIC PLOT
                        #*****************************
                        self.axs[which_plot].text( x, 
                                                   y, 
                                                   str(idx) )
                    
                        
        for which_plot in PLOTS:

            #******************************************************************
            # INVERTING Y AXIS IN ORDER TO KEEP COHERENCE WITH Y AXIS IN OPENCV
            #******************************************************************
            if self.config.OPENCV_COORD_SYS and ( not self.axs[which_plot].yaxis_inverted() ):
                
                self.axs[which_plot].invert_yaxis()


#%%

    def plot_modified_grid( self,
                            MAIN_LINE  = True,
                            STRIP_LINE = True,
                            PLOTS      = [ 'grid2data_dynamic' ] ):
        
        
        #****************************************************
        #CHECK IF THE GRID WAS BUILT BEFORE TRYING TO PLOT IT
        #****************************************************
        if self.grid_is_built:
            
            
            #***************************
            # PLOTTING GLOBAL MAIN LINES
            #***************************
            if MAIN_LINE:
                
                for x_data, y_data in zip( self.modified_x_global_main_grid_lines, 
                                           self.modified_y_global_main_grid_lines ):
                
                    for which_plot in PLOTS:

                        self.axs[which_plot].plot( x_data,
                                                   y_data,
                                                   color     = self.config.MAIN_LINE_GRID_COLOR,
                                                   linestyle = 'dashed',
                                                   linewidth = 1.5  )
                
                
            #****************************
            # PLOTTING GLOBAL STRIP LINES
            #****************************
            if STRIP_LINE:
            
                for x_data, y_data in zip( self.modified_x_global_strip_grid_lines, 
                                           self.modified_y_global_strip_grid_lines ):
                
                    for which_plot in PLOTS:

                        self.axs[which_plot].plot( x_data,
                                                   y_data,
                                                   color     = self.config.STRIP_LINE_GRID_COLOR,
                                                   linestyle = 'dashed',
                                                   linewidth = 1.5  )    
                                
                        
        #*****************************************************************
        # PLOTTING LINES PASSING THROUGH THE CENTER OF GRID ROTATION POINT 
        #*****************************************************************
        for which_plot in PLOTS:
            
            self.axs[which_plot].axvline( x = self.rotation_axis_grid[0], linestyle = 'dashed' )
            self.axs[which_plot].axhline( y = self.rotation_axis_grid[1], linestyle = 'dashed' )            

        
        for which_plot in PLOTS:

            #******************************************************************
            # INVERTING Y AXIS IN ORDER TO KEEP COHERENCE WITH Y AXIS IN OPENCV
            #******************************************************************
            if self.config.OPENCV_COORD_SYS and ( not self.axs[which_plot].yaxis_inverted() ):
                
                self.axs[which_plot].invert_yaxis()
                
                
        #*********************************************************
        # UPDATING AXES LIMITS
        #*********************************************************
        self.update_axes_limits()
        

        #*********************************************************
        # SETTING AXES LIMITS
        #*********************************************************
        self.set_axes_limits()
                

#%%

    def plot_modified_data( self,
                            AS_LINES      = True,
                            COLOR         = 'blue',
                            INCLUDE_LABEL = True,
                            PLOTS         = [ 'data2grid_dynamic' ] ):
        

        if AS_LINES:
            
            #*********************************
            # PLOTTING DATA USING SCATTER PLOT
            #*********************************
            for x_data, y_data in zip( self.modified_x_input_data.reshape( (-1, 2) ), 
                                       self.modified_y_input_data.reshape( (-1, 2) ) ):
                
                for which_plot in PLOTS:

                    self.axs[which_plot].plot( x_data, 
                                               y_data,
                                               color = COLOR )

            
        else:
            
            
            for which_plot in PLOTS:

                #*******************************************
                # PLOTTING DATA USING SCATTER ON STATIC PLOT
                #*******************************************
                self.axs[which_plot].scatter( self.modified_x_input_data, 
                                              self.modified_y_input_data,
                                              color = COLOR )
                
                
        #**************************************
        # INSERTING LABELS ON INPUT DATA POINTS
        #**************************************
        if INCLUDE_LABEL:
            
            if AS_LINES:
                
                #*********************************
                # PLOTTING DATA USING SCATTER PLOT
                #*********************************
                for idx, (x, y) in enumerate( zip( self.modified_x_input_data.reshape( (-1, 2) ), 
                                                   self.modified_y_input_data.reshape( (-1, 2) ) ), 1 ):
    
    
                    for which_plot in PLOTS:

                        self.axs[which_plot].text( x[0], 
                                                   y[0], 
                                                   str(idx) )          
                    
                
            else:
                
                for idx, (x, y) in enumerate( zip( self.modified_x_input_data, self.modified_y_input_data ), 0 ):
                    
                    for which_plot in PLOTS:

                        #*****************************
                        # PLOTTING DATA ON STATIC PLOT
                        #*****************************
                        self.axs[which_plot].text( x, 
                                                   y, 
                                                   str(idx) )
                    

        #*****************************************************************
        # PLOTTING LINES PASSING THROUGH THE CENTER OF DATA ROTATION POINT 
        #*****************************************************************
        for which_plot in PLOTS:
            
            self.axs[which_plot].axvline( x = self.rotation_axis_data[0], linestyle = 'dashed' )
            self.axs[which_plot].axhline( y = self.rotation_axis_data[1], linestyle = 'dashed' )            

                        
        for which_plot in PLOTS:

            #******************************************************************
            # INVERTING Y AXIS IN ORDER TO KEEP COHERENCE WITH Y AXIS IN OPENCV
            #******************************************************************
            if self.config.OPENCV_COORD_SYS and ( not self.axs[which_plot].yaxis_inverted() ):
                
                self.axs[which_plot].invert_yaxis()
                
                
        #*********************************************************
        # UPDATING AXES LIMITS
        #*********************************************************
        self.update_axes_limits()
        

        #*********************************************************
        # SETTING AXES LIMITS
        #*********************************************************
        self.set_axes_limits()
        

#%%

    def rotate_data( self,
                     x_data,
                     y_data,
                     rot_angle_deg,
                     rotation_axis = None ):
        
        if rotation_axis is None:
        
            self.rotation_axis_data = [ 0.0, 0.0 ] 
            
        else:
            
            self.rotation_axis_data = rotation_axis
            
            
        rot_angle_rad = np.deg2rad( rot_angle_deg )

        rotationMatrix = np.array( [ [ np.cos( rot_angle_rad ), -np.sin( rot_angle_rad ) ],
                                     [ np.sin( rot_angle_rad ),  np.cos( rot_angle_rad ) ] ] )
        
        
        stacked_xy_data = np.column_stack( [ x_data, y_data ] )
        
        rotated_xy_data = ( ( stacked_xy_data - self.rotation_axis_data) @ rotationMatrix + self.rotation_axis_data )
            
        
        #*****************************
        # RETURNING ROTATED INPUT DATA
        #*****************************
        return rotated_xy_data[:, 0], rotated_xy_data[:, 1]
        

#%%

    def translate_data( self,
                        x_data,
                        y_data,
                        delta_x,
                        delta_y ):
        
        
        traslateMatrix = np.array( [ [1, 0, delta_x], 
                                     [0, 1, delta_y], 
                                     [0, 0,       1] ] )

        stacked_xy_data = np.column_stack( [ x_data.flatten(), y_data.flatten(), np.ones_like( y_data.flatten() ) ] )
        
        traslated_xy_data = ( traslateMatrix @ stacked_xy_data.T ).T
        
        
        #*******************************
        # RETURNING TRASLATED INPUT DATA
        #*******************************
        return traslated_xy_data[:, 0], traslated_xy_data[:, 1]


#%%

    def modify_data( self,
                     delta_x,
                     delta_y,
                     rot_angle_deg,
                     rotation_axis,
                     sequence ):
        
        if sequence == 'RT':
            
            rotated_x_data, rotated_y_data     = self.rotate_data( x_data        = self.x_input_data,
                                                                   y_data        = self.y_input_data,
                                                                   rot_angle_deg = rot_angle_deg )
            
            
            traslated_x_data, traslated_y_data = self.translate_data( x_data  = rotated_x_data, 
                                                                      y_data  = rotated_y_data, 
                                                                      delta_x = delta_x, 
                                                                      delta_y = delta_y )
            
            self.modified_x_input_data = traslated_x_data
            self.modified_y_input_data = traslated_y_data
        
        elif sequence == 'TR':
            
            
            traslated_x_data, traslated_y_data = self.translate_data( x_data  = self.x_input_data, 
                                                                      y_data  = self.y_input_data, 
                                                                      delta_x = delta_x, 
                                                                      delta_y = delta_y )
            
            rotated_x_data, rotated_y_data     = self.rotate_data( x_data        = traslated_x_data,
                                                                   y_data        = traslated_y_data,
                                                                   rot_angle_deg = rot_angle_deg )

            self.modified_x_input_data = rotated_x_data
            self.modified_y_input_data = rotated_y_data
            
    
#%%

    def rotate_grid( self,
                     x_data,
                     y_data,
                     rot_angle_deg,
                     rotation_axis = None ):
        
        
        if rotation_axis is None:
        
            self.rotation_axis_grid = [ 0.0, 0.0 ] 
            
        else:
            
            self.rotation_axis_grid = rotation_axis
            
            
        rot_angle_rad = np.deg2rad( rot_angle_deg )

        rotateMatrix = np.array( [ [ np.cos( rot_angle_rad ), -np.sin( rot_angle_rad ) ],
                                   [ np.sin( rot_angle_rad ),  np.cos( rot_angle_rad ) ] ] )
    
        

        #***********************************************
        # ROTATING GLOBAL MAIN GRID LINES
        #***********************************************
        stacked_xy_grid = np.column_stack( [ x_data[0].flatten(), 
                                             y_data[0].flatten() ] )
        
        rotated_xy_global_main_grid_lines = ( ( stacked_xy_grid - self.rotation_axis_grid) @ rotateMatrix + self.rotation_axis_grid )
        
        

        #***********************************************
        # ROTATING GLOBAL STRIP GRID LINES
        #***********************************************
        stacked_xy_grid = np.column_stack( [ x_data[1].flatten(), 
                                             y_data[1].flatten() ] )
        
        rotated_xy_global_strip_grid_lines = ( ( stacked_xy_grid - self.rotation_axis_grid) @ rotateMatrix + self.rotation_axis_grid )


        #***********************************
        # RETURNING ROTATED INPUT GRID LINES
        #***********************************
        return ( rotated_xy_global_main_grid_lines[:, 0].reshape(-1, 2),  rotated_xy_global_main_grid_lines[:, 1].reshape(-1, 2),
                 rotated_xy_global_strip_grid_lines[:, 0].reshape(-1, 2), rotated_xy_global_strip_grid_lines[:, 1].reshape(-1, 2) )

        
#%%

    def translate_grid( self,
                        x_data,
                        y_data,
                        delta_x,
                        delta_y ):
        
        
        trasnlateMatrix = np.array( [ [1, 0, delta_x], 
                                      [0, 1, delta_y], 
                                      [0, 0,       1] ] )

        #***********************************************
        # TRASLATING GLOBAL MAIN GRID LINES
        #***********************************************
        stacked_xy_grid = np.column_stack( [ x_data[0].flatten(), 
                                             y_data[0].flatten(),
                                             np.ones_like( y_data[0].flatten() ) ] )
        
        
        translated_xy_global_main_grid_lines = ( trasnlateMatrix @ stacked_xy_grid.T ).T
        
        
        
        #************************************************
        # TRASLATING GLOBAL STRIP GRID LINES
        #************************************************
        stacked_xy_grid = np.column_stack( [ x_data[1].flatten(), 
                                             y_data[1].flatten(),
                                             np.ones_like( y_data[1].flatten() ) ] )


        translated_xy_global_strip_grid_lines = ( trasnlateMatrix @ stacked_xy_grid.T ).T


        #**************************************
        # RETURNING TRANSLATED INPUT GRID LINES
        #**************************************
        return ( translated_xy_global_main_grid_lines[:, 0],  translated_xy_global_main_grid_lines[:, 1],
                 translated_xy_global_strip_grid_lines[:, 0], translated_xy_global_strip_grid_lines[:, 1] )


#%%

    def modify_grid( self,
                     delta_x,
                     delta_y,
                     rot_angle_deg,
                     rotation_axis,
                     sequence ):
        
        if sequence == 'RT':
            
            ( rotated_x_global_main_grid_lines,  rotated_y_global_main_grid_lines,
              rotated_x_global_strip_grid_lines, rotated_y_global_strip_grid_lines ) = self.rotate_grid( x_data       = [ self.x_data_main_grid_lines, self.x_data_strip_grid_lines ],
                                                                                                         y_data       = [ self.y_data_main_grid_lines, self.y_data_strip_grid_lines ],
                                                                                                         rot_angle_deg = rot_angle_deg )
            
            
            ( translated_x_global_main_grid_lines,  translated_y_global_main_grid_lines,
              translated_x_global_strip_grid_lines, translated_y_global_strip_grid_lines ) = self.translate_grid( x_data  = [ rotated_x_global_main_grid_lines, rotated_x_global_strip_grid_lines ], 
                                                                                                                  y_data  = [ rotated_y_global_main_grid_lines, rotated_y_global_strip_grid_lines ], 
                                                                                                                  delta_x = delta_x, 
                                                                                                                  delta_y = delta_y )
            
            self.modified_x_global_main_grid_lines  = translated_x_global_main_grid_lines
            self.modified_y_global_main_grid_lines  = translated_y_global_main_grid_lines
            self.modified_x_global_strip_grid_lines = translated_x_global_strip_grid_lines
            self.modified_y_global_strip_grid_lines = translated_y_global_strip_grid_lines
        
        elif sequence == 'TR':
    
            ( translated_x_global_main_grid_lines,  translated_y_global_main_grid_lines,
              translated_x_global_strip_grid_lines, translated_y_global_strip_grid_lines ) = self.translate_grid( x_data  = [ self.x_data_main_grid_lines, self.x_data_strip_grid_lines ], 
                                                                                                                  y_data  = [ self.y_data_main_grid_lines, self.y_data_strip_grid_lines ],
                                                                                                                  delta_x = delta_x, 
                                                                                                                  delta_y = delta_y )
            
            ( rotated_x_global_main_grid_lines,  rotated_y_global_main_grid_lines,
              rotated_x_global_strip_grid_lines, rotated_y_global_strip_grid_lines ) = self.rotate_grid( x_data       = [ translated_x_global_main_grid_lines, translated_x_global_strip_grid_lines ],
                                                                                                         y_data       = [ translated_y_global_main_grid_lines, translated_y_global_strip_grid_lines ],
                                                                                                         rot_angle_deg = rot_angle_deg )
            

            self.modified_x_global_main_grid_lines  = rotated_x_global_main_grid_lines
            self.modified_y_global_main_grid_lines  = rotated_y_global_main_grid_lines
            self.modified_x_global_strip_grid_lines = rotated_x_global_strip_grid_lines
            self.modified_y_global_strip_grid_lines = rotated_y_global_strip_grid_lines
            
            
#%%

    def count_lines_inside_grid( self ):
        
        
        self.num_lines_inside_grid = 0
        
        self.idx_lines_inside_grid.clear()
        
        
        # lines_inside_grid = []
        
        
        #**************************************
        # LOOPING THROUGH ALL INPUT DATA POINTS
        #**************************************
        for index, (points_x, points_y) in enumerate( zip( self.modified_x_input_data.reshape(-1, 2), 
                                                           self.modified_y_input_data.reshape(-1, 2) ), 1 ):
            
            
            #*****************************************************************
            # CHECKING IF A INPUT DATA POINT IS INSIDE THE GLOBAL BOUNDING BOX
            #*****************************************************************
            if all(  ( ( self.bounding_box[0][0]  <= points_x ) & ( points_x <= self.bounding_box[0][-1]  ) ) & 
                     ( ( self.bounding_box[-1][0] <= points_y ) & ( points_y <= self.bounding_box[-1][-1] ) ) ):
                
                
                
                
                #*****************************************************************************************************
                # CHECKING IF A INPUT DATA POINT WHICH IS INSIDE THE GLOBAL BOUNDING BOX IS ALSO IN ANY VERTICAL STRIP
                #*****************************************************************************************************
                if any( all( (leftmost <= points_x) & (points_x <= rightmost) ) for (leftmost, rightmost) in self.vertical_strip_lines_ends):
                    
                    self.num_lines_inside_grid += 1
                    
                    self.idx_lines_inside_grid.append( index )
    
                    # lines_inside_grid.append( [ points_x, points_y])
                    
                    continue
                    
    
                #*******************************************************************************************************
                # CHECKING IF A INPUT DATA POINT WHICH IS INSIDE THE GLOBAL BOUNDING BOX IS ALSO IN ANY HORIZONTAL STRIP
                #*******************************************************************************************************
                if any( all( (lower <= points_y) & (points_y <= upper) ) for (lower, upper) in self.horizontal_strip_lines_ends):
                    
                    self.num_lines_inside_grid += 1
                
                    self.idx_lines_inside_grid.append( index )

                    # lines_inside_grid.append( [ points_x, points_y])
                    
                    continue
                
                    
        # for x_data, y_data in lines_inside_grid:
            
        #     for which_plot in [ 'initial_static', 'initial_dynamic' ]:

        #         self.axs[which_plot].plot( x_data,
        #                                    y_data,
        #                                    color = 'red' )
        

#%%

    def run( self,
             ACTIVITIES = [],
             DELAY      = False ):
        
        
        #************************************************************************************************
        # CREATE INITIAL POPULATION (GENERATION 0)
        #************************************************************************************************
        population        = self.toolbox.population(n = self.config.POPULATION_SIZE )
        generationCounter = 0
        global_maxFitness = 0
        global_avgFitness = 0


        #************************************************************************************************
        # MAIN EVOLUTIONARY LOOP WHICH STOPS IF THE NUMBER OF GENERATIONS EXCEEDED THE PRESET VALUE
        #************************************************************************************************
        while generationCounter < self.config.MAX_GENERATIONS:
        
            
            #************************************************************************************************
            # UPDATE GENERATION COUNTER
            #************************************************************************************************
            generationCounter += 1
            
            
            #************************************************************************************************
            # APPLY THE SELECTION OPERATOR, TO SELECT THE NEXT GENERATION'S INDIVIDUALS
            #************************************************************************************************
            offspring = self.toolbox.select( population, len( population ) )
            
            
            #************************************************************************************************
            # CLONE THE SELECTED INDIVIDUALS
            #************************************************************************************************
            offspring = list( map( self.toolbox.clone, offspring ) )
            
            
            #************************************************************************************************
            # APPLY THE CROSSOVER OPERATOR TO PAIRS OF OFFSPRING
            #************************************************************************************************
            for child1, child2 in zip( offspring[::2], offspring[1::2] ):
                
                if random.random() < self.config.P_CROSSOVER:
                    
                    self.toolbox.mate(child1, child2)
                    
                    del child1.fitness.values
                    del child2.fitness.values
                    
    
            for mutant in offspring:
                
                if random.random() < self.config.P_MUTATION:
                    
                    self.toolbox.mutate(mutant)
                    
                    del mutant.fitness.values
                    
                    
            #************************************************************************************************
            # CALCULATE FITNESS FOR THE INDIVIDUALS WITH NO PREVIOUS CALCULATED FITNESS VALUE
            #************************************************************************************************
            freshIndividuals   = [ ind for ind in offspring if not ind.fitness.valid ]
            freshFitnessValues = list( map(self.toolbox.evaluate, freshIndividuals) )
            for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
                
                individual.fitness.values = fitnessValue
                
                
            #************************************************************************************************
            # REPLACE THE CURRENT POPULATION WITH THE OFFSPRING
            #************************************************************************************************
            population[:] = offspring
            
            
            #************************************************************************************************
            # COLLECT FITNESSVALUES INTO A LIST, UPDATE STATISTICS AND PRINT
            #************************************************************************************************
            fitnessValues = [ind.fitness.values[0] for ind in population]
            
            
            current_maxFitness = max(fitnessValues)
            current_avgFitness = sum(fitnessValues) / len(population)
            
            self.maxFitnessValues.append( current_maxFitness )
            self.meanFitnessValues.append( current_avgFitness )
            
            
            #print("- Generation [ {0:5d}/{1:5d} ]: \n\tMax Fitness.......= {2:8.4f}, \n\tAvg Fitness.......= {3:8.4f},".format( generationCounter,
            #                                                                                                                    self.config.MAX_GENERATIONS,
            #                                                                                                                    current_maxFitness, 
            #                                                                                                                    current_avgFitness ) )


            #************************************************************************************************
            # FIND AND PRINT BEST INDIVIDUAL
            #************************************************************************************************
            best_index = fitnessValues.index( max(fitnessValues) )
            self.best_individual_history.append( population[ best_index ] )
            
            if ( ( current_maxFitness > global_maxFitness ) or 
                 ( current_avgFitness > global_avgFitness )  ):
            
                global_maxFitness = current_maxFitness
                global_avgFitness = current_avgFitness
                
                self.best_individual = population[ best_index ]
                
                
            #print("\tBest Individual...= ", *[ '{0:8.4f}'.format(item) for item in population[ best_index ] ], "\n")  
            
            
            #*********************************
            # PERFORMING ACTIVITIES LIST TASKS
            #*********************************
            #if ACTIVITIES:            


            #    self.update_plots( ACTIVITIES, 
            #                       DELAY )
                
        
        
        
        return [ self.best_individual[0], self.best_individual[1], self.best_individual[2], global_maxFitness, global_avgFitness ]
        
                
#%%

    def plot_best_individual( self,
                              AS_LINES      = True,
                              COLOR         = 'blue',
                              INCLUDE_LABEL = True,
                              MAIN_LINE     = True,
                              STRIP_LINE    = True ):

        
        #**********************************************************
        # ------------------PLOTTING INITIAL DATA------------------
        #**********************************************************
        
        PLOTS = [ 'grid2data' ]
        if AS_LINES:
            
            #*********************************
            # PLOTTING DATA USING SCATTER PLOT
            #*********************************
            for x_data, y_data in zip( self.x_input_data.reshape( (-1, 2) ), 
                                       self.y_input_data.reshape( (-1, 2) ) ):
                
                for which_plot in [ 'grid2data' ]:

                    self.axs_final[which_plot].plot( x_data, 
                                                     y_data,
                                                     color = COLOR )

            
        else:
            
            
            for which_plot in PLOTS:

                #*******************************************
                # PLOTTING DATA USING SCATTER ON STATIC PLOT
                #*******************************************
                self.axs_final[which_plot].scatter( self.x_input_data, 
                                                    self.y_input_data,
                                                    color = COLOR )
            
        
        #**************************************
        # INSERTING LABELS ON INPUT DATA POINTS
        #**************************************
        if INCLUDE_LABEL:
            
            if AS_LINES:
                
                #*********************************
                # PLOTTING DATA USING SCATTER PLOT
                #*********************************
                for idx, (x, y) in enumerate( zip( self.x_input_data.reshape( (-1, 2) ), 
                                                   self.y_input_data.reshape( (-1, 2) ) ), 1 ):
    
    
                    for which_plot in PLOTS:

                        self.axs_final[which_plot].text( x[0], 
                                                         y[0],
                                                         str(idx) )           
                    
                
            else:
                
                for idx, (x, y) in enumerate( zip( self.x_input_data, self.y_input_data ), 0 ):
                    
                    for which_plot in PLOTS:

                        #*****************************
                        # PLOTTING DATA ON STATIC PLOT
                        #*****************************
                        self.axs_final[which_plot].text( x, 
                                                         y, 
                                                         str(idx) )
                    
                        
        for which_plot in PLOTS:

            #******************************************************************
            # INVERTING Y AXIS IN ORDER TO KEEP COHERENCE WITH Y AXIS IN OPENCV
            #******************************************************************
            if self.config.OPENCV_COORD_SYS and ( not self.axs_final[which_plot].yaxis_inverted() ):
                
                self.axs_final[which_plot].invert_yaxis()
                

        #*************************************************************
        # -------------------PLOTTING MODIFIED DATA-------------------
        #*************************************************************

        PLOTS = [ 'data2grid' ]
        
        
        # *****************************
        # ROTATING AND TRANSLATING DATA
        # *****************************
        self.modify_data( delta_x       = self.best_individual[0],
                          delta_y       = self.best_individual[1],
                          rot_angle_deg = self.best_individual[2],
                          rotation_axis = [0.0, 0.0],
                          sequence      = 'RT' )
        
        
        # ********************************************
        # PLOTTING ROTATED AND TRANSLATED INITIAL DATA
        # ********************************************
        if AS_LINES:
            
            #*********************************
            # PLOTTING DATA USING SCATTER PLOT
            #*********************************
            for x_data, y_data in zip( self.modified_x_input_data.reshape( (-1, 2) ), 
                                       self.modified_y_input_data.reshape( (-1, 2) ) ):
                
                for which_plot in PLOTS:

                    self.axs_final[which_plot].plot( x_data, 
                                                     y_data,
                                                     color = COLOR )

            
        else:
            
            
            for which_plot in PLOTS:

                #*******************************************
                # PLOTTING DATA USING SCATTER ON STATIC PLOT
                #*******************************************
                self.axs_final[which_plot].scatter( self.modified_x_input_data, 
                                                    self.modified_y_input_data,
                                                    color = COLOR )
                
                
        #**************************************
        # INSERTING LABELS ON INPUT DATA POINTS
        #**************************************
        if INCLUDE_LABEL:
            
            if AS_LINES:
                
                #*********************************
                # PLOTTING DATA USING SCATTER PLOT
                #*********************************
                for idx, (x, y) in enumerate( zip( self.modified_x_input_data.reshape( (-1, 2) ), 
                                                   self.modified_y_input_data.reshape( (-1, 2) ) ), 1 ):
    
    
                    for which_plot in PLOTS:

                        self.axs_final[which_plot].text( x[0], 
                                                         y[0], 
                                                         str(idx) )          
                    
                
            else:
                
                for idx, (x, y) in enumerate( zip( self.modified_x_input_data, self.modified_y_input_data ), 0 ):
                    
                    for which_plot in PLOTS:

                        #*****************************
                        # PLOTTING DATA ON STATIC PLOT
                        #*****************************
                        self.axs_final[which_plot].text( x, 
                                                         y, 
                                                         str(idx) )
                    

        #*****************************************************************
        # PLOTTING LINES PASSING THROUGH THE CENTER OF DATA ROTATION POINT 
        #*****************************************************************
        for which_plot in PLOTS:
            
            self.axs_final[which_plot].axvline( x = self.rotation_axis_data[0], linestyle = 'dashed' )
            self.axs_final[which_plot].axhline( y = self.rotation_axis_data[1], linestyle = 'dashed' )            

                        
        for which_plot in PLOTS:

            #******************************************************************
            # INVERTING Y AXIS IN ORDER TO KEEP COHERENCE WITH Y AXIS IN OPENCV
            #******************************************************************
            if self.config.OPENCV_COORD_SYS and ( not self.axs_final[which_plot].yaxis_inverted() ):
                
                self.axs_final[which_plot].invert_yaxis()
                
                
        #**********************************************************
        # ------------------PLOTTING INITIAL GRID------------------
        #**********************************************************
        
        PLOTS = [ 'data2grid' ]
        #****************************************************
        #CHECK IF THE GRID WAS BUILT BEFORE TRYING TO PLOT IT
        #****************************************************
        if self.grid_is_built:
            
            
            if MAIN_LINE:
                
                for x_data, y_data in zip( self.x_data_main_grid_lines, 
                                           self.y_data_main_grid_lines ):
                
                    for which_plot in PLOTS:
                        
                        self.axs_final[which_plot].plot( x_data,
                                                         y_data,
                                                         color     = self.config.MAIN_LINE_GRID_COLOR,
                                                         linestyle = 'dashed',
                                                         linewidth = 1.5  )
                    
                
            if STRIP_LINE:
            
                for x_data, y_data in zip( self.x_data_strip_grid_lines, 
                                           self.y_data_strip_grid_lines ):
                
                    for which_plot in PLOTS:

                        self.axs_final[which_plot].plot( x_data,
                                                         y_data,
                                                         color     = self.config.STRIP_LINE_GRID_COLOR,
                                                         linestyle = 'dashed',
                                                         linewidth = 1.5  )    
                                    
        for which_plot in PLOTS:
            
            
            #******************************************************************
            # INVERTING Y AXIS IN ORDER TO KEEP COHERENCE WITH Y AXIS IN OPENCV
            #******************************************************************
            if self.config.OPENCV_COORD_SYS and ( not self.axs_final[which_plot].yaxis_inverted() ):
                
                self.axs[which_plot].invert_yaxis()
                
                
        #*************************************************************
        # -------------------PLOTTING MODIFIED GRID-------------------
        #*************************************************************
        
        PLOTS = [ 'grid2data' ]
        
        
        # *****************************
        # ROTATING AND TRANSLATING GRID
        # *****************************
        self.modify_grid( delta_x       = -self.best_individual[0],
                          delta_y       = -self.best_individual[1],
                          rot_angle_deg = -self.best_individual[2],
                          rotation_axis =  self.rotation_axis_data,
                          sequence      = 'TR' )

        
        #****************************************************
        #CHECK IF THE GRID WAS BUILT BEFORE TRYING TO PLOT IT
        #****************************************************
        if self.grid_is_built:
            
            
            #***************************
            # PLOTTING GLOBAL MAIN LINES
            #***************************
            if MAIN_LINE:
                
                for x_data, y_data in zip( self.modified_x_global_main_grid_lines, 
                                           self.modified_y_global_main_grid_lines ):
                
                    for which_plot in PLOTS:

                        self.axs_final[which_plot].plot( x_data,
                                                         y_data,
                                                         color     = self.config.MAIN_LINE_GRID_COLOR,
                                                         linestyle = 'dashed',
                                                         linewidth = 1.5  )
                
                
            #****************************
            # PLOTTING GLOBAL STRIP LINES
            #****************************
            if STRIP_LINE:
            
                for x_data, y_data in zip( self.modified_x_global_strip_grid_lines, 
                                           self.modified_y_global_strip_grid_lines ):
                
                    for which_plot in PLOTS:

                        self.axs_final[which_plot].plot( x_data,
                                                         y_data,
                                                         color     = self.config.STRIP_LINE_GRID_COLOR,
                                                         linestyle = 'dashed',
                                                         linewidth = 1.5  )    
                                
                        
        #*****************************************************************
        # PLOTTING LINES PASSING THROUGH THE CENTER OF GRID ROTATION POINT 
        #*****************************************************************
        for which_plot in PLOTS:
            
            self.axs_final[which_plot].axvline( x = self.rotation_axis_grid[0], linestyle = 'dashed' )
            self.axs_final[which_plot].axhline( y = self.rotation_axis_grid[1], linestyle = 'dashed' )            

        
        for which_plot in PLOTS:

            #******************************************************************
            # INVERTING Y AXIS IN ORDER TO KEEP COHERENCE WITH Y AXIS IN OPENCV
            #******************************************************************
            if self.config.OPENCV_COORD_SYS and ( not self.axs_final[which_plot].yaxis_inverted() ):
                
                self.axs_final[which_plot].invert_yaxis()


                
            
#%%

    def update_plots( self,
                      ACTIVITIES,
                      DELAY = 0 ):

        #******************************
        # CLEARING AXES AFTER REDRAWING
        #******************************
        PLOTS = [ 'initial_static', 'data2grid_dynamic', 'grid2data_dynamic' ]
        for which_plot in PLOTS:
            
            
            self.axs[which_plot].clear()
            self.axs[which_plot].set_xlim( self.x_limits )
            self.axs[which_plot].set_ylim( self.y_limits )
            
            
            #*******************
            # UPDATING IN A LOOP
            #*******************
            self.fig.canvas.draw() 
            self.fig.canvas.flush_events() 
        
        
        # ********************
        # PLOTTING INPUT DATA
        # ********************
        self.plot_initial_data( AS_LINES      = True,
                                COLOR         = 'blue',
                                INCLUDE_LABEL = True )
        
        
        # *********************
        # PLOTTING INITIAL GRID
        # *********************
        self.plot_initial_grid( MAIN_LINE  = True,
                                STRIP_LINE = True )


        for activity in ACTIVITIES:
            
                
            
            if activity.lower() == 'data':
                
                # *****************************
                # ROTATING AND TRANSLATING DATA
                # *****************************
                self.modify_data( delta_x       = self.best_individual[0],
                                  delta_y       = self.best_individual[1],
                                  rot_angle_deg = self.best_individual[2],
                                  rotation_axis = [0.0, 0.0],
                                  sequence      = 'RT' )
                
                
                # ********************************************
                # PLOTTING ROTATED AND TRANSLATED INITIAL DATA
                # ********************************************
                self.plot_modified_data( AS_LINES      = True,
                                         COLOR         = 'blue',
                                         INCLUDE_LABEL = True )                       
                                    

            if activity.lower() == 'grid':
                
                # *****************************
                # ROTATING AND TRANSLATING GRID
                # *****************************
                self.modify_grid( delta_x       = -self.best_individual[0],
                                  delta_y       = -self.best_individual[1],
                                  rot_angle_deg = -self.best_individual[2],
                                  rotation_axis =  self.rotation_axis_data,
                                  sequence      = 'TR' )
                
                
                # ********************************************
                # PLOTTING ROTATED AND TRANSLATED INITIAL GRID
                # ********************************************
                self.plot_modified_grid( MAIN_LINE  = True,
                                         STRIP_LINE = True )
                

        #*********************
        # UPDATING AXES LIMITS
        #*********************
        self.update_axes_limits()
        

        #*********************
        # SETTING AXES LIMITS
        #*********************
        self.set_axes_limits()


        #*******************
        # UPDATING IN A LOOP
        #*******************
        self.fig.canvas.draw() 
        self.fig.canvas.flush_events() 
        
        
        #********************************
        # MAKING A SLEEP OF DELAY SECONDS
        #********************************
        if DELAY:
            
            time.sleep( DELAY )
        
        
#%%
