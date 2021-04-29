# ----------------------------------------------Import required Modules----------------------------------------------- #

import os

# ----------------------------------------------Set File Paths-------------------------------------------------------- #

# Colab
# TAXONOMY_FILE_PATH    = '/content/drive/My Drive/3DR/src/ShapeNet_mid_2_other.json'
# RENDERING_PATH        = '/content/drive/My Drive/3DR/Datasets/ShapeNetRendering/{}/{}/rendering'
# VOXEL_PATH            = '/content/drive/My Drive/3DR/Datasets/ShapeNetVox32/{}/{}/model.binvox'

# Dhruv
# Training
TAXONOMY_FILE_PATH    = 'E:\\Projects\\3D_Reconstruction\\src\\ShapeNet_mid_2_other.json'
RENDERING_PATH        = 'E:\\Datasets\\3D_Reconstruction\ShapeNetRendering\\{}\\{}\\rendering'
VOXEL_PATH            = 'E:\\Datasets\\3D_Reconstruction\ShapeNetVox32\\{}\\{}\\model.binvox'
# Inference
# RENDERING_PATH      = os.path.join('E:\\Datasets\\3D_Reconstruction\Inference\Rendered Image', os.listdir('E:\\Datasets\\3D_Reconstruction\Inference\Rendered Image')[0])
# GROUND_TRUTH_PATH   = os.path.join('E:\\Datasets\\3D_Reconstruction\Inference\Ground Truth', os.listdir('E:\\Datasets\\3D_Reconstruction\Inference\Ground Truth')[0])
# VOXEL_SAVE_PATH     = os.path.join('E:\\Datasets\\3D_Reconstruction\Inference')


# Suraj
# Training
# TAXONOMY_FILE_PATH    = 'C:\\Users\\bidnu\\Documents\\Suraj_Docs\\3D_Project\\ShapeNet_P2V\\ShapeNet.json'
# RENDERING_PATH        = 'C:\\Users\\bidnu\\Documents\\Suraj_Docs\\3D_Project\\ShapeNet_P2V\\ShapeNetRendering\\{}\\{}\\rendering'
# VOXEL_PATH            = 'C:\\Users\\bidnu\\Documents\\Suraj_Docs\\3D_Project\\ShapeNet_P2V\\ShapeNetVox32\\{}\\{}\\model.binvox'
# Inference
# RENDERING_PATH        = ""
# GROUND_TRUTH_PATH     = ""
# VOXEL_SAVE_PATH       = ""

# Rishab
# Training
# TAXONOMY_FILE_PATH    = 'D:\\P2V\\Pix2Vox-master\\datasets\\ShapeNet_test_v3.json'
# RENDERING_PATH        = 'D:\\ShapeNet_new\\ShapeNetRendering\\{}\\{}\\rendering'
# VOXEL_PATH            = 'D:\\ShapeNet_new\\ShapeNetVox32\\{}\\{}\\model.binvox'
# Inference
# RENDERING_PATH        = ""
# GROUND_TRUTH_PATH     = ""
# VOXEL_SAVE_PATH       = ""

# VM
# TAXONOMY_FILE_PATH      = '/home/drs/3D_Project/src/ShapeNet_original.json'
# RENDERING_PATH          = '/home/drs/3D_Project/datasets/ShapeNet/ShapeNetRendering/{}/{}/rendering'
# VOXEL_PATH              = '/home/drs/3D_Project/datasets/ShapeNet/ShapeNetVox32/{}/{}/model.binvox'


# ----------------------------------------------Training Configuration------------------------------------------------ #

input_shape = (224, 224, 3)  # input shape
autoencoder_flavour = "vanilla" # Vanilla or Variational AutoEncoder
encoder_cnn = "vgg" # pre-trained encoder cnn (vgg, resnet or densenet)
batch_size = 2  # batch size
epochs = 10 #250 # Number of epochs
learning_rate = 0.001 # Learning rate
boundaries = [150] # Boundary epoch for learning rate scheduler
model_save_frequency = 1 #10 # Save model every n epochs (specify n)
checkpoint_path = os.path.join(os.getcwd(), 'saved_models') # Model save path

restrict_dataset = True
restriction_size = 200