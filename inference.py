# ----------------------------------------------Import required Modules----------------------------------------------- #

import os
import glob
import datetime
import argparse
import numpy

import tensorflow as tf

from tqdm import tqdm

import config as cfg
from logger import logger_test
import data
import metrics as metr
import save_data
import utils

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
    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation.
    with tf.GradientTape() as tape:
        # Run the forward pass of the layer.
        # The operations that the layer applies
        # to its inputs are going to be recorded
        # on the GradientTape.

        logits = autoencoder_model(x, training=False)  # Logits for this minibatch

        # Compute the loss value for this minibatch.
        loss_value = loss_fn(y, logits)

    return loss_value, logits


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
        x_test = tf.reshape(
            tensor = x_test, shape = (-1, 224, 224, 3)
        )
        y_test = tf.reshape(
            tensor=y_test, shape=(-1, 32, 32, 32)
        )
        logger.info("render path -> ", x_test.shape)
        logger.info("gt path -> ", y_test.shape)

        test_loss, logits = compute_train_metrics(x_test, y_test)
        iou = metr.calc_iou_loss(y_test, logits)

        # print(test_loss, iou)

    logger.info("Inference loss -> {0}".format(test_loss))
    logger.info("Inference IoU -> {0}".format(iou))

    logger.info("End of program execution")