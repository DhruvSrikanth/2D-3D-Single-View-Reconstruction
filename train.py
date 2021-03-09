# ----------------------------------------------Import required Modules----------------------------------------------- #

import os
import glob
import datetime
import argparse

import tensorflow as tf

from tqdm import tqdm

import config as cfg
import logger as log
import data
import model
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
parser.add_argument('--epochs', type=int, default=cfg.epochs, help='Number of epochs.')
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
epochs = args.epochs
learning_rate = args.lr
boundaries = cfg.boundaries
model_save_frequency = args.model_save_frequency
checkpoint_path = args.checkpoint_path

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

    # Get train path lists and train data generator
    train_path_list = data.get_xy_paths(taxonomy_dict=taxonomy_dict, rendering_path=RENDERING_PATH, voxel_path=VOXEL_PATH, mode='train')
    train_path_list_sample = train_path_list[:100] + train_path_list[-100:]  # just for testing purposes
    train_dataset = tf.data.Dataset.from_generator(data.tf_data_generator, args=[train_path_list_sample], output_types = (tf.float32, tf.float32, tf.string))
    train_dataset = train_dataset.batch(batch_size).shuffle(150).prefetch(tf.data.AUTOTUNE)

    # Get validation path lists and validation data generator
    val_path_list = data.get_xy_paths(taxonomy_dict=taxonomy_dict, rendering_path=RENDERING_PATH, voxel_path=VOXEL_PATH, mode='val')
    val_path_list_sample = val_path_list[:20] + val_path_list[-20:]  # just for testing purposes
    val_dataset = tf.data.Dataset.from_generator(data.tf_data_generator, args=[val_path_list_sample],output_types = (tf.float32, tf.float32, tf.string))
    val_dataset = val_dataset.batch(batch_size).shuffle(150).prefetch(tf.data.AUTOTUNE)

    # Get test path lists and test data generator
    test_path_list = data.get_xy_paths(taxonomy_dict=taxonomy_dict, rendering_path=RENDERING_PATH, voxel_path=VOXEL_PATH, mode='test')
    test_path_list_sample = test_path_list[:20] + test_path_list[-20:]  # just for testing purposes
    test_dataset = tf.data.Dataset.from_generator(data.tf_data_generator, args=[test_path_list_sample],output_types = (tf.float32, tf.float32, tf.string))
    test_dataset = test_dataset.batch(batch_size).shuffle(150).prefetch(tf.data.AUTOTUNE)

    # Load Model and Resume Training, otherwise Start Training

    # Check if model save path exists
    if not os.path.isdir(checkpoint_path):
        print("\nNo model save directory found...\nCreating model save directory at - ", checkpoint_path)
        os.mkdir(checkpoint_path)
    else:
        print("\nFound model save directory at - ", checkpoint_path)

    saved_model_files = glob.glob(checkpoint_path + "\*.h5")
    if len(saved_model_files) == 0:
        resume_epoch = 0
        autoencoder_model = model.build_autoencoder(input_shape, describe=True)
        print("\nStarting Training")
    else:
        latest_model = os.path.join(checkpoint_path, saved_model_files[-1])
        autoencoder_model = tf.keras.models.load_model(latest_model, compile=False)
        resume_epoch = int(latest_model.split("_")[-1].split(".")[0])
        print("\nResuming Training On Epoch -> ", resume_epoch + 1)
        print("\nLoading Model From -> ", latest_model)

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

    # tensorboard writer for testing values
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # dictionary holds mean iou of each class for training data
    mean_iou_train = dict()

    # dictionary holds mean iou of each class for validation data
    mean_iou_val = dict()

    # dictionary holds mean iou of each class for testing data
    mean_iou_test = dict()

    # Training Loop
    num_training_samples = len(train_path_list_sample)
    num_validation_steps = len(val_path_list_sample) // batch_size
    num_test_steps = len(test_path_list_sample) // batch_size

    # bar_val = progressbar.ProgressBar(maxval=num_validation_samples).start()
    # bar_test = progressbar.ProgressBar(maxval=num_test_samples).start()

    end_epoch = epochs
    for epoch in range(resume_epoch, end_epoch, 1):
        print("\nepoch {}/{}".format(epoch + 1, end_epoch))

        learning_rate = learning_rate_fn(epoch)

        progBar = tf.keras.utils.Progbar(num_training_samples, stateful_metrics=['loss_fn'], verbose=1)

        iou_dict = dict()

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train, tax_id) in enumerate(train_dataset):
            tax_id = tax_id.numpy()
            tax_id = [item.decode("utf-8") for item in tax_id] # byte string (b'hello' to regular string 'hello')

            train_loss, logits = compute_train_metrics(x_batch_train, y_batch_train)

            iou = op.calc_iou_loss(y_batch_train, logits)

            iou_dict = op.iou_dict_update(tax_id, iou_dict, iou)
            mean_iou_train = op.calc_mean_iou(iou_dict, mean_iou_train)

            values=[('train_loss', train_loss)]

            progBar.add(batch_size, values)

        allClass_mean_iou = sum(mean_iou_train.values()) / len(mean_iou_train)
      
        print("training iou: {}".format(mean_iou_train))
        print("all class mean training iou: {}". format(allClass_mean_iou))

        # TODO: Training and Validation Loss -> 1 graph, Training and Validation IOU (mean IOU over all classes)
        with train_summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss, step=epoch)
            tf.summary.scalar('train_iou_plane', allClass_mean_iou, step=epoch)

        # Iterate over the batches of the dataset and calculate validation loss
        log.log_status(3,"Validation phase for epoch-{} running now".format(epoch+1))
        for step, (x_batch_val, y_batch_val, tax_id) in tqdm(enumerate(val_dataset), total=num_validation_steps):
            tax_id = tax_id.numpy()
            tax_id = [item.decode("utf-8") for item in tax_id] # byte string (b'hello' to regular string 'hello')

            val_loss, logits = compute_train_metrics(x_batch_val, y_batch_val)

            iou = op.calc_iou_loss(y_batch_val, logits)

            # IoU dict update moved to iou_dict_update function
            iou_dict = op.iou_dict_update(tax_id, iou_dict, iou)
            mean_iou_val = op.calc_mean_iou(iou_dict, mean_iou_val)

        allClass_mean_iou = sum(mean_iou_val.values()) / len(mean_iou_val)

        with val_summary_writer.as_default():
            tf.summary.scalar('val_loss', val_loss, step=epoch)
            tf.summary.scalar('val_iou_plane', allClass_mean_iou, step=epoch)
        
        print("validation iou: {}".format(mean_iou_val))
        print("all class mean training iou: {}". format(allClass_mean_iou))

        # Save Model During Training
        if (epoch+1) % model_save_frequency == 0:
            model_save_file = 'ae_model_epoch_{}.h5'.format(epoch+1)
            model_save_file_path = os.path.join(checkpoint_path, model_save_file)
            print("Saving Autoencoder Model at ", model_save_file_path)
            tf.keras.models.save_model(model=autoencoder_model, filepath=model_save_file_path, overwrite=True, include_optimizer=True)

    log.log_status(3, "Testing phase running now")
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
            tf.summary.scalar('test_iou_plane', allClass_mean_iou, step=step)
    
    print("testing iou: {}".format(mean_iou_test))
    print("all class mean training iou: {}". format(allClass_mean_iou))

    log.log_status(3, "\nEnd of program execution")