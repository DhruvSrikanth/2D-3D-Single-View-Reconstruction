# ----------------------------------------------Import required Modules----------------------------------------------- #

import os

# ----------------------------------------------Set File Paths-------------------------------------------------------- #

TAXONOMY_FILE_PATH    = 'E:\\Projects\\3D_Reconstruction\\3DR_src\\ShapeNet.json'
RENDERING_PATH        = 'E:\\Datasets\\3D_Reconstruction\ShapeNetRendering\\{}\\{}\\rendering'
VOXEL_PATH            = 'E:\\Datasets\\3D_Reconstruction\ShapeNetVox32\\{}\\{}\\model.binvox'

# TAXONOMY_FILE_PATH    = 'C:\\Users\\bidnu\\Documents\\Suraj_Docs\\3D_Project\\ShapeNet_P2V\\ShapeNet.json'
# RENDERING_PATH        = 'C:\\Users\\bidnu\\Documents\\Suraj_Docs\\3D_Project\\ShapeNet_P2V\\ShapeNetRendering\\{}\\{}\\rendering'
# VOXEL_PATH            = 'C:\\Users\\bidnu\\Documents\\Suraj_Docs\\3D_Project\\ShapeNet_P2V\\ShapeNetVox32\\{}\\{}\\model.binvox'

# ----------------------------------------------Training Configuration------------------------------------------------ #

input_shape = (224, 224, 3)  # input shape
batch_size = 1  # batch size
epochs = 4  # Number of epochs
learning_rate = 0.001 # Learning rate
boundaries = [150] # Boundary epoch for learning rate scheduler
model_save_frequency = 2 # Save model every n epochs (specify n)
checkpoint_path = os.path.join(os.getcwd(), 'saved_models') # Model save path