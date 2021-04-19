# ----------------------------------------------Import required Modules----------------------------------------------- #

import os
import glob
import datetime
import argparse

import tensorflow as tf

from tqdm import tqdm

import config as cfg
from logger import logger_train
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

parser.add_argument('--taxonomy_path', type=str, default=cfg.TAXONOMY_FILE_PATH, help='Taxonomy file path.')
parser.add_argument('--render_path', type=str, default=cfg.RENDERING_PATH, help='Rendering images path.')
parser.add_argument('--voxel_path', type=str, default=cfg.VOXEL_PATH, help='Voxel models path.')

parser.add_argument('--ae_flavour', type=str, default=cfg.autoencoder_flavour, help='AutoEncoder model architecture.')
parser.add_argument('--encoder_cnn', type=str, default=cfg.encoder_cnn, help='Pre trained encoder model architecture.')
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
encoder_cnn = args.encoder_cnn
batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.lr
boundaries = cfg.boundaries
model_save_frequency = args.model_save_frequency
checkpoint_path = args.checkpoint_path
autoencoder_flavour = args.ae_flavour

# ----------------------------------------------Set Logger------------------------------------------------------------ #

logger = logger_train

# -------------------------------------------Print Training Params---------------------------------------------------- #

logger.info("Input Shape -> {0}\n Batch Size -> {1}\n Epochs -> {2}\n Learning Rate -> {3}".format(input_shape, batch_size, epochs, learning_rate))

# -------------------------------------------Print Model Params------------------------------------------------------- #

logger.info("AutoEncoder Flavour -> {0}\n Encoder Block Type -> {1}\n ".format(autoencoder_flavour, encoder_cnn))

# -----------------------------Define Loss, Optimizer & Compute Metrics Function-------------------------------------- #

# Loss
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Optimizer
opt = tf.keras.optimizers.Adam()

@tf.function
def compute_train_metrics(x, y, opt, mode="Train"):
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

        if mode == "train":
            # Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss, autoencoder_model.trainable_weights)
            # Run one step of gradient descent by updating the value of the variables to minimize the loss.
            opt.apply_gradients(zip(grads, autoencoder_model.trainable_weights))

    return loss, x_logits

# ----------------------------------------------Run Main Code--------------------------------------------------------- #

restrict_dataset = True
restriction_size = 100

if __name__ == '__main__':

    # Read Data

    # Get train path lists and train data generator
    if restrict_dataset:
        train_DataLoader = data.DataLoader(TAXONOMY_FILE_PATH, RENDERING_PATH, VOXEL_PATH, "train", batch_size, restrict=restrict_dataset, restriction_size=restriction_size)
    else:
        train_DataLoader = data.DataLoader(TAXONOMY_FILE_PATH, RENDERING_PATH, VOXEL_PATH, "train", batch_size)
    train_data_gen = train_DataLoader.dataset_gen


    # Get validation path lists and validation data generator
    if restrict_dataset:
        val_DataLoader = data.DataLoader(TAXONOMY_FILE_PATH, RENDERING_PATH, VOXEL_PATH, "val", batch_size, restrict=restrict_dataset, restriction_size=restriction_size)
    else:
        val_DataLoader = data.DataLoader(TAXONOMY_FILE_PATH, RENDERING_PATH, VOXEL_PATH, "val", batch_size)
    val_data_gen = val_DataLoader.dataset_gen

    # Load Model and Resume Training, otherwise Start Training
    # Check if model save path exists
    if not os.path.isdir(checkpoint_path):
        # print("\nNo model save directory found...\nCreating model save directory at - ", checkpoint_path)
        logger.info("No model save directory found, so creating directory at -> {0}".format(checkpoint_path))
        os.mkdir(checkpoint_path)
    else:
        # print("\nFound model save directory at - ", checkpoint_path)
        logger.info("Found model save directory at -> {0}".format(checkpoint_path))

    saved_model_files = os.listdir(checkpoint_path)
    # saved_model_files = utils.model_sort(saved_model_files)
    # print(saved_model_files)
    if len(saved_model_files) == 0:
        resume_epoch = 0
        autoencoder_model = model.AutoEncoder(custom_input_shape=tuple([-1] + list(input_shape)), ae_flavour=autoencoder_flavour, enc_net=encoder_cnn)
        logger.info("Starting Training phase")
    else:
        latest_model = os.path.join(checkpoint_path, saved_model_files[-1])
        # autoencoder_model = tf.keras.models.load_model(latest_model, compile=False)
        autoencoder_model = model.AutoEncoder(custom_input_shape=tuple([-1] + list(input_shape)), ae_flavour=autoencoder_flavour, enc_net=encoder_cnn)
        temp_tensor = tf.zeros((8,224,224,3), dtype=tf.dtypes.float32)
        reconstruction = autoencoder_model(temp_tensor)
        del temp_tensor
        del reconstruction
        autoencoder_model.load_weights(latest_model)
        resume_epoch = int(latest_model.split("_")[-1].split(".")[0])
        logger.info("Resuming Training on Epoch -> {0}".format(resume_epoch + 1))
        logger.info("Loading Model from -> {0}".format(latest_model))

    # Learning Rate Scheduler
    # learning rate becomes 0.01*0.5 after 150 epochs else it is 0.01*1.0
    values = [learning_rate, learning_rate * 0.5]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

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
    num_training_samples = train_DataLoader.length
    num_validation_steps = val_DataLoader.length // batch_size

    end_epoch = resume_epoch + epochs
    for epoch in range(resume_epoch, end_epoch, 1):
        print("\nepoch {}/{}".format(epoch + 1, end_epoch))

        learning_rate = learning_rate_fn(epoch)

        progBar = tf.keras.utils.Progbar(num_training_samples, stateful_metrics=['loss_fn'], verbose=1)

        iou_dict = dict()

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train, tax_id) in enumerate(train_data_gen):

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
        utils.record_iou_data(1, epoch + 1, mean_iou_train)

        # TODO: Training and Validation Loss -> 1 graph, Training and Validation IOU (mean IOU over all classes)
        with train_summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss, step=epoch)
            tf.summary.scalar('overall_train_iou', allClass_mean_iou, step=epoch)

        # Save Loss value in CSV file
        utils.record_loss(1, epoch+1,train_loss.numpy())

        # Iterate over the batches of the dataset and calculate validation loss
        logger.info("Validation phase running now for Epoch - {0}".format(epoch + 1))
        # for step, (x_batch_val, y_batch_val, tax_id) in tqdm(enumerate(val_dataset), total=num_validation_steps):
        for step, (x_batch_val, y_batch_val, tax_id) in tqdm(enumerate(val_data_gen), total=num_validation_steps):
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
        utils.record_iou_data(2, epoch + 1, mean_iou_val)

        # Save Model During Training
        if (epoch + 1) % model_save_frequency == 0:
            model_save_file = autoencoder_flavour + 'AutoEncoder_{0}_model_epoch_{1}.h5'.format(encoder_cnn, epoch + 1)
            model_save_file_path = os.path.join(checkpoint_path, model_save_file)
            logger.info("Saving {0} Autoencoder Model at {1}".format("V" + autoencoder_flavour.lower()[1:], model_save_file_path))
            # tf.keras.models.save_model(model=autoencoder_model, filepath=model_save_file_path, overwrite=True, include_optimizer=True)
            # autoencoder_model.save(model_save_file_path, save_format='tf')
            autoencoder_model.save_weights(model_save_file_path, overwrite=True)
    logger.info("End of program execution")