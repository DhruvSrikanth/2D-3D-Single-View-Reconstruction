# ----------------------------------------------Import required Modules----------------------------------------------- #

import os
import glob
import datetime
import argparse
import numpy as np
import cv2

import tensorflow as tf

from tqdm import tqdm

import config as cfg
from logger import logger_test
import data
import metrics as metr
import save_data
import utils
import binvox_rw as bin_rw
import binvox_viz as bin_viz

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

# ----------------------------------------------Set Logger------------------------------------------------------------ #

logger = logger_test

# ----------------------------------------------Inference Function---------------------------------------------------- #

# Compute loss
@tf.function
def compute_train_metrics(x, y):
    '''
    Compute training metrics for custom training loop.\n
    :param x: input to model\n
    :param y: output from model\n
    :return: training metrics i.e loss
    '''
    # Open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation.
    with tf.GradientTape() as tape:
        # Run the forward pass of the layer. The operations that the layer applies to its inputs are going to be recorded on the GradientTape.
        # Logits for this minibatch
        x_logits = autoencoder_model(x, training=True)
        # Compute the loss value for this minibatch.
        loss = loss_fn(y, x_logits)
        loss += sum(autoencoder_model.losses)

    return loss, x_logits


# ----------------------------------------------Run Main Code--------------------------------------------------------- #

if __name__ == '__main__':

    # Read Data
    inference_paths = [RENDERING_PATH, GROUND_TRUTH_PATH]
    test_dataset = data.tf_data_generator(inference_paths, 'Inference')

    # Load Model for Inference phase
    # Check if model save path exists
    if not os.path.isdir(checkpoint_path):
        # logger.error("No saved model found. Please run train.py to train a model and save it for Testing purposes")
        exit()
    else:
        saved_model_files = glob.glob(checkpoint_path + "\*.h5")
        saved_model_files = utils.model_sort(saved_model_files)
        if len(saved_model_files) == 0:
            logger.error("No saved model found. Please run train.py to train a model and save it for Testing purposes")
            exit()
        else:
            logger.info("Found model save directory at -> {0}".format(checkpoint_path))
            pass

    saved_model_files = glob.glob(checkpoint_path + "\*.h5")
    latest_model = os.path.join(checkpoint_path, saved_model_files[-1])
    autoencoder_model = tf.keras.models.load_model(latest_model, compile=False)
    print(autoencoder_model.summary())

    logger.info("Loading Model from -> {0}".format(latest_model))

    # Loss
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # logger.info("Testing phase running now")
    for x_test, y_test in test_dataset:
        x_test = tf.reshape(tensor = x_test, shape = (-1, 224, 224, 3))
        y_test = tf.reshape(tensor=y_test, shape=(-1, 32, 32, 32))

        test_loss, logits = compute_train_metrics(x_test, y_test)
        iou = metr.calc_iou_loss(y_test, logits)
        iou_val = iou[0]

    logger.info("Inference loss -> {0}".format(test_loss))
    logger.info("Inference IoU -> {0}".format(iou_val))

    # Save Voxel Model
    gv_ = logits.numpy()
    gv = np.squeeze(gv_)
    voxel_model_save_fp = VOXEL_SAVE_PATH + '\\voxel_model.binvox'
    sample_path = 'E:\\Datasets\\3D_Reconstruction\\Inference\\Ground Truth\\model.binvox'
    bin_rw.np_binvox(gv, voxel_model_save_fp, sample_path)

    # returned_image = bin_viz.get_volume_views(gv, VOXEL_SAVE_PATH)

    logger.info("End of program execution")