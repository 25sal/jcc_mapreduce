import semantic_segmentation.Configuration                             as Config
import segmentation_models_pytorch               as smp
import segmentation_models_pytorch.utils.metrics
import numpy                                     as np
import pathlib
import pandas                                    as pd
import cv2
import PIL
 

from semantic_segmentation.Model            import UNet_Model
from semantic_segmentation.System_Manager   import System_Manager
  



def map2mask( image_file,
              mask_file ):
	

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
    
    
	#**************************
	# INITIALIZE MODEL
	#**************************
	unet_model = UNet_Model


	#**************************
	# DEFINE LOSS FUNCTION
	#**************************
	criterion = smp.utils.losses.DiceLoss()


	#***************
	# DEFINE METRICS
	#***************
	metrics = [ smp.utils.metrics.IoU(threshold = 0.5) ]


	#***************************
	# INSTANTIATE SYSTEM MANAGER
	#***************************
	Manager = System_Manager( model           = unet_model,
		          	   trainDataLoader = None,
		          	   validDataLoader = None,
		                  testDataLoader  = None,
		          	   criterion       = criterion,
		          	   optimizer       = None,
		          	   metrics         = metrics,
		                  config          = Config )
		                  
		                  

	#************************
	# READ IMAGES AND MASKS
	#************************
	image = cv2.cvtColor( cv2.imread( str( image_file ) ), cv2.COLOR_BGR2RGB )
	print(image.shape)
	image = np.transpose(image, (2, 0, 1))
	print(image.shape)


	#********************
	#PERFORM LOADING STEP
	#********************
	predicted_mask = Manager.predict( image, select_class_rgb_values, LOAD_MODEL = True )
	
	
	#salvataggio maschera
	#*******************************************************************
	# SAVING ORIGINAL MASK USING PATH DEFINED IN THE CONFIGURATION FILE
	#*******************************************************************
	cv2.imwrite( str( mask_file ), mask )
		                  
		                  
if __name__ == '__main__':

	image_file = ""
	map2mask(image_file, mask_file)
		                  
		                  
		                  
