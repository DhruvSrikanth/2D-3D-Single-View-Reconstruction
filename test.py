# ----------------------------------------------Import required Modules----------------------------------------------- #

import os
import glob
import datetime
import argparse

import tensorflow as tf

from tqdm import tqdm

import config as cfg
from logger import logger_test
import data
import model
import metrics as metr
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

parser.add_argument('--taxonomy_path', type=str, default=cfg.TAXONOMY_FILE_PATH, help='Specify the taxonomy file path.')
parser.add_argument('--render_path', type=str, default=cfg.RENDERING_PATH, help='Specify the rendering images path.')
parser.add_argument('--voxel_path', type=str, default=cfg.VOXEL_PATH, help='Specify the voxel models path.')
parser.add_argument('--batch_size', type=int, default=cfg.batch_size, help='Batch size.')
parser.add_argument('--checkpoint_path', type=str, default=cfg.checkpoint_path, help='Start training from existing models.')

args = parser.parse_args()

# ----------------------------------------------Set File Paths-------------------------------------------------------- #

TAXONOMY_FILE_PATH    = args.taxonomy_path
RENDERING_PATH        = args.render_path
VOXEL_PATH            = args.voxel_path

# ----------------------------------------------Testing Configuration------------------------------------------------ #

input_shape = cfg.input_shape
batch_size = args.bs
checkpoint_path = args.checkpoint_path

# ----------------------------------------------Set Logger------------------------------------------------------------ #

logger = logger_test

# ----------------------------------------------Test Function-------------------------------------------------------- #

# Compute loss
@tf.function
def compute_train_metrics(x,y):
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

        # TODO: check the training=True parameter in the below function
        logits = autoencoder_model(x, training=False)  # Logits for this minibatch

        # Compute the loss value for this minibatch.
        loss_value = loss_fn(y, logits)

    return loss_value, logits

# ----------------------------------------------Run Main Code--------------------------------------------------------- #

if __name__ == '__main__':

    # Read Data
    test_DataLoader = data.DataLoader(TAXONOMY_FILE_PATH, RENDERING_PATH, VOXEL_PATH, batch_size=batch_size, mode="train")
    test_path_list = test_DataLoader.path_list
    # Load Model for Testing phase
    # Check if model save path exists
    if not os.path.isdir(checkpoint_path):
        logger.error("No saved model found. Please run train.py to train a model and save it for Testing purposes")
        exit()
    else:
        saved_model_files = glob.glob(checkpoint_path + "\*.h5")
        saved_model_files = utils.model_sort(saved_model_files)
        if len(saved_model_files) == 0:
            logger.error("No saved model found. Please run train.py to train a model and save it for Testing purposes")
            exit()
        else:
            logger.info("Found model save directory at -> {0}".format(checkpoint_path))

    saved_model_files = glob.glob(checkpoint_path + "\*.h5")
    latest_model = os.path.join(checkpoint_path, saved_model_files[-1])
    autoencoder_model = model.AutoEncoder(custom_input_shape=tuple([-1] + list(input_shape)), ae_flavour="variational", enc_net="vgg")
    temp_tensor = tf.zeros((8, 224, 224, 3), dtype=tf.dtypes.float32)
    reconstruction = autoencoder_model(temp_tensor)
    del temp_tensor
    del reconstruction
    autoencoder_model.load_weights(latest_model)
    
    logger.info("Loading Model from -> {0}".format(latest_model))

    # Loss
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Tensorboard Graph
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # tensorboard writer for testing values
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # dictionary holds mean iou of each class for testing data
    mean_iou_test = dict()

    # Training Loop
    num_test_steps = test_DataLoader.length // batch_size

    iou_dict = dict()

    logger.info("Testing phase running now")
    for step, (x_batch_test, y_batch_test, tax_id) in tqdm(enumerate(test_DataLoader.data_gen(test_DataLoader)), total=num_test_steps):
        tax_id = tax_id.numpy()
        tax_id = [item.decode("utf-8") for item in tax_id] # byte string (b'hello' to regular string 'hello')

        test_loss, logits = compute_train_metrics(x_batch_test, y_batch_test)

        iou = metr.calc_iou_loss(y_batch_test, logits)

        # IoU dict update moved to iou_dict_update function
        iou_dict = metr.iou_dict_update(tax_id, iou_dict, iou)
        mean_iou_test = metr.calc_mean_iou(iou_dict, mean_iou_test)

        allClass_mean_iou = sum(mean_iou_test.values()) / len(mean_iou_test)

        with test_summary_writer.as_default():
            tf.summary.scalar('test_loss', test_loss, step=step)
            tf.summary.scalar('overall_test_iou', allClass_mean_iou, step=step)
    
    logger.info("Testing IoU -> {0}".format(mean_iou_test))
    logger.info("Overall mean Testing IoU -> {0}". format(allClass_mean_iou))

    # Save testing IoU values in CSV file
    utils.record_iou_data(3, step+1,mean_iou_test)

    # Save Loss value in CSV file
    utils.record_loss(2, step+1,test_loss.numpy())

    logger.info("End of program execution")