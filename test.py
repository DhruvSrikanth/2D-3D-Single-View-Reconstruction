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
import optimizer as op

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

parser.add_argument('--bs', type=int, default=cfg.batch_size, help='Batch size.')
parser.add_argument('--lr', type=float, default=cfg.learning_rate, help='Learning rate.')

parser.add_argument('--model_save_frequency', type=int, default=cfg.model_save_frequency, help='Model save frequency.')
parser.add_argument('--checkpoint_path', type=str, default=cfg.checkpoint_path, help='Start training from existing models.')

args = parser.parse_args()

# ----------------------------------------------Set File Paths-------------------------------------------------------- #

TAXONOMY_FILE_PATH    = args.taxonomy_path
RENDERING_PATH        = args.render_path
VOXEL_PATH            = args.voxel_path

# ----------------------------------------------Training Configuration------------------------------------------------ #

input_shape = cfg.input_shape
batch_size = args.bs
learning_rate = args.lr
boundaries = cfg.boundaries
model_save_frequency = args.model_save_frequency
checkpoint_path = args.checkpoint_path

# ----------------------------------------------Set Logger------------------------------------------------------------ #

logger = logger_test

# ----------------------------------------------Train Function-------------------------------------------------------- #

# Custom Train Function
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

        # TODO: check the training parameter in the below function
        logits = autoencoder_model(x, training=True)  # Logits for this minibatch

        # Compute the loss value for this minibatch.
        loss_value = loss_fn(y, logits)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_value, autoencoder_model.trainable_weights)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    opt.apply_gradients(zip(grads, autoencoder_model.trainable_weights))

    return loss_value, logits

# ----------------------------------------------Run Main Code--------------------------------------------------------- #

if __name__ == '__main__':

    # Read Data
    # Read Taxonomy JSON file
    taxonomy_dict = data.read_taxonomy_JSON(TAXONOMY_FILE_PATH)

    # Get test path lists and test data generator
    test_path_list = data.get_xy_paths(taxonomy_dict=taxonomy_dict,
                                       rendering_path=RENDERING_PATH,
                                       voxel_path=VOXEL_PATH,
                                       mode='test')

    test_path_list_sample = test_path_list[:20] + test_path_list[-20:]  # just for testing purposes

    test_dataset = tf.data.Dataset.from_generator(data.tf_data_generator,
                                                  args=[test_path_list_sample],
                                                  output_types = (tf.float32, tf.float32, tf.string))

    test_dataset = test_dataset.batch(batch_size).shuffle(150).prefetch(tf.data.AUTOTUNE)

    # Load Model and Resume Training, otherwise Start Training

    # Check if model save path exists
    if not os.path.isdir(checkpoint_path):
        # print("\nNo model save directory found...\nCreating model save directory at - ", checkpoint_path)
        logger.info("No model save directory found, so creating directory at -> {0}".format(checkpoint_path))
        # os.mkdir(checkpoint_path)
        exit()
    else:
        # print("\nFound model save directory at - ", checkpoint_path)
        logger.info("Found model save directory at -> {0}".format(checkpoint_path))

    saved_model_files = glob.glob(checkpoint_path + "\*.h5")
    latest_model = os.path.join(checkpoint_path, saved_model_files[-1])
    autoencoder_model = tf.keras.models.load_model(latest_model, compile=False)
    resume_epoch = int(latest_model.split("_")[-1].split(".")[0])
    # print("\nResuming Training On Epoch -> ", resume_epoch + 1)
    logger.info("Resuming Training on Epoch -> {0}".format(resume_epoch + 1))
    # print("\nLoading Model From -> ", latest_model)
    logger.info("Loading Model from -> {0}".format(latest_model))

    # Loss
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Learning Rate Scheduler
    # learning rate becomes 0.01*0.5 after 150 epochs else it is 0.01*1.0
    values = [learning_rate, learning_rate * 0.5]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

    # Optimizer
    opt = tf.keras.optimizers.Adam()

    # Tensorboard Graph
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # tensorboard writer for testing values
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # dictionary holds mean iou of each class for testing data
    mean_iou_test = dict()

    # Training Loop
    num_test_steps = len(test_path_list_sample) // batch_size

    # bar_test = progressbar.ProgressBar(maxval=num_test_samples).start()

    logger.info("Testing phase running now")
    for step, (x_batch_test, y_batch_test, tax_id) in tqdm(enumerate(test_dataset), total=num_test_steps):
        tax_id = tax_id.numpy()
        tax_id = [item.decode("utf-8") for item in tax_id] # byte string (b'hello' to regular string 'hello')

        test_loss, logits = compute_train_metrics(x_batch_test, y_batch_test)

        iou = op.calc_iou_loss(y_batch_test, logits)

        # IoU dict update moved to iou_dict_update function
        iou_dict = op.iou_dict_update(tax_id, iou_dict, iou)
        mean_iou_test = op.calc_mean_iou(iou_dict, mean_iou_test)

        allClass_mean_iou = sum(mean_iou_test.values()) / len(mean_iou_test)

        with test_summary_writer.as_default():
            tf.summary.scalar('test_loss', test_loss, step=step)
            tf.summary.scalar('overall_test_iou', allClass_mean_iou, step=step)
    
    logger.info("Testing IoU -> {0}".format(mean_iou_test))
    logger.info("Overall mean Testing IoU -> {0}". format(allClass_mean_iou))

    logger.info("End of program execution")