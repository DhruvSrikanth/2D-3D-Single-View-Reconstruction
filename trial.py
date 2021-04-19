# ----------------------------------------------Import required Modules----------------------------------------------- #

import os
import glob
import argparse
import numpy as np

import tensorflow as tf

import config as cfg

import data
import model


# ----------------------------------------------Set Environment Variables--------------------------------------------- #

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# tf.debugging.set_log_device_placement(True)

# the following 2 commands are used to suppress some tf warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel('ERROR')

# ----------------------------------------------Define Command Line Argument Parser----------------------------------- #

# Argument Parser
parser = argparse.ArgumentParser(description='3D Reconstruction Using an Autoencoder via Transfer Learning')

parser.add_argument('--render_path', type=str, default=cfg.RENDERING_PATH, help='Specify the rendering images path.')
parser.add_argument('--ground_truth_path', type=str, default=cfg.GROUND_TRUTH_PATH, help='Specify the ground truth path.')
parser.add_argument('--voxel_save_path', type=str, default=cfg.VOXEL_SAVE_PATH, help='Specify the voxel models path.')
parser.add_argument('--checkpoint_path', type=str, default=cfg.checkpoint_path, help='Start training from existing models.')

args = parser.parse_args()

# ----------------------------------------------Set File Paths-------------------------------------------------------- #

RENDERING_PATH = args.render_path
GROUND_TRUTH_PATH = args.ground_truth_path
VOXEL_SAVE_PATH = args.voxel_save_path

# ----------------------------------------------Testing Configuration------------------------------------------------ #

input_shape = cfg.input_shape
checkpoint_path = args.checkpoint_path
TAXONOMY_FILE_PATH = cfg.TAXONOMY_FILE_PATH

# ----------------------------------------------Run Main Code--------------------------------------------------------- #

if __name__ == '__main__':

    # Read Data
    inference_DataLoader = data.DataLoader(TAXONOMY_FILE_PATH, RENDERING_PATH, GROUND_TRUTH_PATH, mode="inference")
    inference_data_gen = inference_DataLoader.dataset_gen

    encoder = model.Encoder(custom_input_shape=(224, 224, 3), ae_flavour="variational", enc_net="densenet")
    decoder = model.Decoder(ae_flavour="variational")

    # logger.info("Testing phase running now")
    for x_test, y_test in inference_data_gen:
        x_test = tf.reshape(tensor = x_test, shape = (-1, 224, 224, 3))
        y_test = tf.reshape(tensor=y_test, shape=(-1, 32, 32, 32))

        z_mean, z_log_var, z = encoder(x_test)
        x_logits = decoder(z)

        print(x_logits.numpy().shape)
        print(encoder.summary())
        print(decoder.summary())