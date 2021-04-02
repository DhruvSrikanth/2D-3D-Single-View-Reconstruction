# ----------------------------------------------Import required Modules----------------------------------------------- #

import os
import glob
import datetime
import argparse
import numpy

import tensorflow as tf

from tqdm import tqdm

import config as cfg
from logger import logger_train
import data
import model
import metrics as metr
import utils
import saveiou

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
parser.add_argument('--epochs', type=int, default=cfg.epochs, help='Number of epochs.')
parser.add_argument('--lr', type=float, default=cfg.learning_rate, help='Learning rate.')

parser.add_argument('--model_save_frequency', type=int, default=cfg.model_save_frequency, help='Model save frequency.')
parser.add_argument('--checkpoint_path', type=str, default=cfg.checkpoint_path,
                    help='Start training from existing models.')

args = parser.parse_args()

# ----------------------------------------------Set File Paths-------------------------------------------------------- #

TAXONOMY_FILE_PATH = args.taxonomy_path
RENDERING_PATH = args.render_path
VOXEL_PATH = args.voxel_path

# ----------------------------------------------Training Configuration------------------------------------------------ #

input_shape = cfg.input_shape
batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.lr
boundaries = cfg.boundaries
model_save_frequency = args.model_save_frequency
checkpoint_path = args.checkpoint_path

# ----------------------------------------------Set Logger------------------------------------------------------------ #

logger = logger_train

# -------------------------------------------Print Model Params------------------------------------------------------- #

logger.info("image input shape -> {0}\n bacth_size -> {1}\n epochs -> {2}\n learning_rate -> {3}".format(input_shape, batch_size, epochs, learning_rate))

# ----------------------------------------------Train Function-------------------------------------------------------- #

# Compute Loss
# @tf.function(experimental_compile=True)
@tf.function
def compute_train_metrics(x, y, mode="Train"):
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

    if mode == "train":
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

    # Get train path lists and train data generator
    train_path_list = data.get_xy_paths(taxonomy_dict=taxonomy_dict,
                                        rendering_path=RENDERING_PATH,
                                        voxel_path=VOXEL_PATH,
                                        mode='train')

    # train_path_list_sample = train_path_list[:5000] + train_path_list[-5000:]  # just for testing purposes
    train_path_list_sample = train_path_list[:10]

    # train_dataset = tf.data.Dataset.from_generator(data.tf_data_generator,
    #                                                args=[train_path_list_sample],
    #                                                output_types=(tf.float32, tf.float32, tf.string))

    # train_dataset = train_dataset.batch(batch_size).shuffle(150).prefetch(tf.data.AUTOTUNE)

    # Get validation path lists and validation data generator
    val_path_list = data.get_xy_paths(taxonomy_dict=taxonomy_dict,
                                      rendering_path=RENDERING_PATH,
                                      voxel_path=VOXEL_PATH,
                                      mode='val')

    val_path_list_sample = val_path_list #val_path_list[:20] + val_path_list[-20:]  # just for testing purposes

    val_path_list_sample = val_path_list_sample[:10]

    # val_dataset = tf.data.Dataset.from_generator(data.tf_data_generator,
    #                                              args=[val_path_list_sample],
    #                                              output_types=(tf.float32, tf.float32, tf.string))

    # val_dataset = val_dataset.batch(batch_size).shuffle(150).prefetch(tf.data.AUTOTUNE)

    # Load Model and Resume Training, otherwise Start Training

    # Check if model save path exists
    if not os.path.isdir(checkpoint_path):
        # print("\nNo model save directory found...\nCreating model save directory at - ", checkpoint_path)
        logger.info("No model save directory found, so creating directory at -> {0}".format(checkpoint_path))
        os.mkdir(checkpoint_path)
    else:
        # print("\nFound model save directory at - ", checkpoint_path)
        logger.info("Found model save directory at -> {0}".format(checkpoint_path))

    saved_model_files = glob.glob(checkpoint_path + "/*.h5")
    saved_model_files = utils.model_sort(saved_model_files)
    if len(saved_model_files) == 0:
        resume_epoch = 0
        autoencoder_model = model.build_autoencoder(input_shape, describe=False)
        logger.info("Starting Training phase")
    else:
        latest_model = os.path.join(checkpoint_path, saved_model_files[-1])
        autoencoder_model = tf.keras.models.load_model(latest_model, compile=False)
        resume_epoch = int(latest_model.split("_")[-1].split(".")[0])
        logger.info("Resuming Training on Epoch -> {0}".format(resume_epoch + 1))
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

    # tensorboard writer for training values
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # tensorboard writer for validation values
    val_log_dir = 'logs/gradient_tape/' + current_time + '/val'
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    # dictionary holds mean iou of each class for training data
    mean_iou_train = dict()

    # dictionary holds mean iou of each class for validation data
    mean_iou_val = dict()

    # Training Loop
    num_training_samples = len(train_path_list_sample)
    num_validation_steps = len(val_path_list_sample) // batch_size

    end_epoch = resume_epoch + epochs
    for epoch in range(resume_epoch, end_epoch, 1):
        print("\nepoch {}/{}".format(epoch + 1, end_epoch))

        learning_rate = learning_rate_fn(epoch)

        progBar = tf.keras.utils.Progbar(num_training_samples, stateful_metrics=['loss_fn'], verbose=1)

        iou_dict = dict()

        # Iterate over the batches of the dataset.
        # for step, (x_batch_train, y_batch_train, tax_id) in enumerate(train_dataset):
        for step, (x_batch_train, y_batch_train, tax_id) in enumerate(data.data_gen(train_path_list_sample, batch_size)):
            # tax_id = tax_id.numpy()
            # tax_id = [item.decode("utf-8") for item in tax_id]  # byte string (b'hello' to regular string 'hello')

            train_loss, logits = compute_train_metrics(x_batch_train, y_batch_train, "train")

            iou = metr.calc_iou_loss(y_batch_train, logits)

            iou_dict = metr.iou_dict_update(tax_id, iou_dict, iou)
            mean_iou_train = metr.calc_mean_iou(iou_dict, mean_iou_train)

            values = [('train_loss', train_loss)]

            progBar.add(batch_size, values)

        allClass_mean_iou = sum(mean_iou_train.values()) / len(mean_iou_train)

        logger.info("Training IoU -> {0}".format(mean_iou_train))
        logger.info("Overall mean Training IoU -> {0}".format(allClass_mean_iou))

        # Save training IoU values in CSV file
        saveiou.record_iou_data_train(epoch + 1, mean_iou_train)

        # TODO: Training and Validation Loss -> 1 graph, Training and Validation IOU (mean IOU over all classes)
        with train_summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss, step=epoch)
            tf.summary.scalar('overall_train_iou', allClass_mean_iou, step=epoch)

        # Save Loss value in CSV file
        saveiou.record_training_loss(epoch+1,train_loss.numpy())

        # Iterate over the batches of the dataset and calculate validation loss
        logger.info("Validation phase running now for Epoch - {0}".format(epoch + 1))
        # for step, (x_batch_val, y_batch_val, tax_id) in tqdm(enumerate(val_dataset), total=num_validation_steps):
        for step, (x_batch_val, y_batch_val, tax_id) in tqdm(enumerate(data.data_gen(val_path_list_sample, batch_size)), total=num_validation_steps):
            # tax_id = tax_id.numpy()
            # tax_id = [item.decode("utf-8") for item in tax_id]  # byte string (b'hello' to regular string 'hello')

            val_loss, logits = compute_train_metrics(x_batch_val, y_batch_val, "val")

            iou = metr.calc_iou_loss(y_batch_val, logits)

            # IoU dict update moved to iou_dict_update function
            iou_dict = metr.iou_dict_update(tax_id, iou_dict, iou)
            mean_iou_val = metr.calc_mean_iou(iou_dict, mean_iou_val)

        allClass_mean_iou = sum(mean_iou_val.values()) / len(mean_iou_val)

        with val_summary_writer.as_default():
            tf.summary.scalar('val_loss', val_loss, step=epoch)
            tf.summary.scalar('overall_val_iou', allClass_mean_iou, step=epoch)

        logger.info("Validation IoU -> {0}".format(mean_iou_val))
        logger.info("Overall mean Validation IoU -> {0}".format(allClass_mean_iou))

        # Save validation IoU values in CSV file
        saveiou.record_iou_data_val(epoch + 1, mean_iou_val)

        # Save Model During Training
        if (epoch + 1) % model_save_frequency == 0:
            model_save_file = 'ae_model_epoch_{0}.h5'.format(epoch + 1)
            model_save_file_path = os.path.join(checkpoint_path, model_save_file)
            logger.info("Saving Autoencoder Model at {0}".format(model_save_file_path))
            tf.keras.models.save_model(model=autoencoder_model, filepath=model_save_file_path, overwrite=True,
                                       include_optimizer=True)

    logger.info("End of program execution")