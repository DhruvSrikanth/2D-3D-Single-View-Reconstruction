# ----------------------------------------------Import required Modules----------------------------------------------- #

import os

# ----------------------------------------------Set File Paths-------------------------------------------------------- #

# Dhruv
TAXONOMY_FILE_PATH    = 'E:\\Projects\\3D_Reconstruction\\3DR_src\\ShapeNet_original.json' # 'E:\\Projects\\3D_Reconstruction\\3DR_src\\ShapeNet.json'
RENDERING_PATH        = 'E:\\Datasets\\3D_Reconstruction\ShapeNetRendering\\{}\\{}\\rendering'
VOXEL_PATH            = 'E:\\Datasets\\3D_Reconstruction\ShapeNetVox32\\{}\\{}\\model.binvox'

# Suraj
# TAXONOMY_FILE_PATH    = 'C:\\Users\\bidnu\\Documents\\Suraj_Docs\\3D_Project\\ShapeNet_P2V\\ShapeNet.json'
# RENDERING_PATH        = 'C:\\Users\\bidnu\\Documents\\Suraj_Docs\\3D_Project\\ShapeNet_P2V\\ShapeNetRendering\\{}\\{}\\rendering'
# VOXEL_PATH            = 'C:\\Users\\bidnu\\Documents\\Suraj_Docs\\3D_Project\\ShapeNet_P2V\\ShapeNetVox32\\{}\\{}\\model.binvox'

# Rishab
# TAXONOMY_FILE_PATH    = 'D:\\P2V\\Pix2Vox-master\\datasets\\ShapeNet_test_v3.json'
# RENDERING_PATH        = 'D:\\ShapeNet_new\\ShapeNetRendering\\{}\\{}\\rendering'
# VOXEL_PATH            = 'D:\\ShapeNet_new\\ShapeNetVox32\\{}\\{}\\model.binvox'

# ----------------------------------------------Training Configuration------------------------------------------------ #

input_shape = (224, 224, 3)  # input shape
batch_size = 4  # batch size
epochs = 2 # Number of epochs
learning_rate = 0.001 # Learning rate
boundaries = [150] # Boundary epoch for learning rate scheduler
model_save_frequency = 1 # Save model every n epochs (specify n)
checkpoint_path = os.path.join(os.getcwd(), 'saved_models') # Model save path