# ----------------------------------------------Import required Modules----------------------------------------------- #

import os
import datetime
import argparse

import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121

from tqdm import tqdm

import config as cfg
from logger import logger_train
import data
import metrics as metr
import utils

# ----------------------------------------------Set Environment Variables--------------------------------------------- #

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Mixed precision optimization
os.environ['TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32'] = '1'
os.environ['TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32'] = '1'
os.environ['TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32'] = '1'
os.environ['export TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
# policy = tf.keras.mixed_precision.Policy('mixed_float16')
# tf.keras.mixed_precision.set_global_policy(policy)

# Auto-clustering
run_file_name = 'model_train' + '.py'
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 " + os.path.join(os.getcwd(), run_file_name)

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

logger.info("\nInput Shape -> {0}\n Batch Size -> {1}\n Epochs -> {2}\n Learning Rate -> {3}".format(input_shape, batch_size, epochs, learning_rate))

# -------------------------------------------Print Model Params------------------------------------------------------- #

logger.info("\nAutoEncoder Flavour -> {0}\n Encoder Block Type -> {1}\n ".format(autoencoder_flavour, encoder_cnn))

# ----------------------------------------------Define Sampling For Latent Space-------------------------------------- #

class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    @tf.function(experimental_compile=True)
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# ----------------------------------------------Define Model---------------------------------------------------------- #

class Encoder(tf.keras.Model):

    def __init__(self, custom_input_shape=(224, 224, 3), ae_flavour="vanilla", enc_net="vgg", latent_dim=128):
        super(Encoder, self).__init__(name="Encoder")
        # Variables
        self.custom_input_shape = custom_input_shape
        self._input_shape = self.custom_input_shape
        self.ae_flavour = ae_flavour.lower()
        self.enc_net = enc_net.lower()
        self.latent_dim = latent_dim
        # CNN-Block
        if self.enc_net == "vgg":
            self.cnn_model = VGG16(include_top=False, weights="imagenet", input_shape=self.custom_input_shape,
                                   pooling="none")
            self.cnn_model.trainable = False
            self.pre_trained = tf.keras.models.Model(inputs=self.cnn_model.input,
                                                     outputs=self.cnn_model.get_layer(name="block4_conv2").output,
                                                     name="vgg")

        elif self.enc_net == "resnet":
            self.cnn_model = ResNet50(include_top=False, weights="imagenet", input_shape=self.custom_input_shape,
                                      pooling="none")
            self.cnn_model.trainable = False
            self.pre_trained = tf.keras.models.Model(inputs=self.cnn_model.input,
                                                     outputs=self.cnn_model.get_layer(name="conv3_block1_out").output,
                                                     name="resnet")

        elif self.enc_net == "densenet":
            self.cnn_model = DenseNet121(include_top=False, weights="imagenet", input_shape=self.custom_input_shape,
                                         pooling="none")
            self.cnn_model.trainable = False
            self.pre_trained = tf.keras.models.Model(inputs=self.cnn_model.input,
                                                     outputs=self.cnn_model.get_layer(name="pool3_relu").output,
                                                     name="densenet")

        # Encoder-Block
        self.layer10 = tf.keras.layers.Conv2D(filters=512, kernel_size=1,
                                              name="conv10")  # for pix2vox-A(large), kernel_size is 3
        self.layer10_norm = tf.keras.layers.BatchNormalization(name="layer10_norm")

        self.layer11 = tf.keras.layers.Conv2D(filters=256, kernel_size=3,
                                              name="conv11")  # for pix2vox-A(large), filters is 512
        self.layer11_norm = tf.keras.layers.BatchNormalization(name="layer11_norm")
        self.layer11_pool = tf.keras.layers.MaxPooling2D(pool_size=(4, 4),
                                                         name="layer11_pool")  # for pix2vox-A(large), kernel size is 3

        self.layer12 = tf.keras.layers.Conv2D(filters=128, kernel_size=3,
                                              name="conv12")  # for pix2vox-A(large), filters is 256, kernel_size is 1
        self.layer12_norm = tf.keras.layers.BatchNormalization(name="layer12_norm")

        if self.ae_flavour == "variational":
            self.bottleneck_encoding = tf.keras.layers.Flatten()

            self.z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")
            self.z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")
            self.sampled = Sampling()

    def call(self, inputs, training=False):

        cnn_block = self.pre_trained(inputs=inputs, training=training)

        layer10_out = self.layer10(cnn_block)
        layer10_norm_out = self.layer10_norm(layer10_out)
        layer10_elu = tf.keras.activations.elu(layer10_norm_out)

        layer11_out = self.layer11(layer10_elu)
        layer11_norm_out = self.layer11_norm(layer11_out)
        layer11_elu = tf.keras.activations.elu(layer11_norm_out)
        layer11_pool_out = self.layer11_pool(layer11_elu)

        layer12_out = self.layer12(layer11_pool_out)
        layer12_norm_out = self.layer12_norm(layer12_out)
        layer12_elu = tf.keras.activations.elu(layer12_norm_out)

        if self.ae_flavour == "variational":
            bottleneck_encoded = self.bottleneck_encoding(layer12_elu)

            z_mean = self.z_mean(bottleneck_encoded)
            z_log_var = self.z_log_var(bottleneck_encoded)
            z = self.sampled((z_mean, z_log_var))

            return z_mean, z_log_var, z

        elif self.ae_flavour == "vanilla":
            return layer12_elu

    def build_graph(self):
        dummy = tf.keras.Input(shape=self.custom_input_shape, name = "Encoder Input")
        enc_model = tf.keras.Model(inputs=[dummy], outputs=self.call(dummy))
        enc_model._name  = "Encoder"
        return enc_model

    def summary(self):
        enc_model = self.build_graph()
        return enc_model.summary()

class Decoder(tf.keras.Model):

    def __init__(self, ae_flavour="vanilla", custom_input_shape = (2, 2, 2, 256)):
        super(Decoder, self).__init__(name="Decoder")
        # Variables
        self.custom_input_shape = custom_input_shape
        self.ae_flavour = ae_flavour.lower()

        if self.ae_flavour == "variational":
            self.layer0 = tf.keras.layers.Dense(2*2*2*256, activation="relu")

        self.layer0_reshape = tf.keras.layers.Reshape((2, 2, 2, 256))

        self.layer1 = tf.keras.layers.Convolution3DTranspose(filters=128, kernel_size=4, strides=(2, 2, 2),
                                                             padding="same", use_bias=False, name="Conv3D_1")
        self.layer1_norm = tf.keras.layers.BatchNormalization(name="layer1_norm")

        self.layer2 = tf.keras.layers.Convolution3DTranspose(filters=64, kernel_size=4, strides=(2, 2, 2),
                                                             padding="same", use_bias=False, name="Conv3D_2")
        self.layer2_norm = tf.keras.layers.BatchNormalization(name="layer2_norm")

        self.layer3 = tf.keras.layers.Convolution3DTranspose(filters=32, kernel_size=4, strides=(2, 2, 2),
                                                             padding="same", use_bias=False, name="Conv3D_3")
        self.layer3_norm = tf.keras.layers.BatchNormalization(name="layer3_norm")

        self.layer4 = tf.keras.layers.Convolution3DTranspose(filters=8, kernel_size=4, strides=(2, 2, 2),
                                                             padding="same", use_bias=False, name="Conv3D_4")
        self.layer4_norm = tf.keras.layers.BatchNormalization(name="layer4_norm")

        self.layer5 = tf.keras.layers.Convolution3DTranspose(filters=1, kernel_size=1, use_bias=False, name="Conv3D_5")
        self.layer5_reshape = tf.keras.layers.Reshape((32, 32, 32))

    def call(self, inputs):

        if self.ae_flavour == "variational":
            layer0_out = self.layer0(inputs)
            layer1_in = self.layer0_reshape(layer0_out)

        elif self.ae_flavour == "vanilla":
            layer1_in = self.layer0_reshape(inputs)

        layer1_out = self.layer1(layer1_in)
        layer1_norm_out = self.layer1_norm(layer1_out)
        layer1_relu = tf.keras.activations.relu(layer1_norm_out)

        layer2_out = self.layer2(layer1_relu)
        layer2_norm_out = self.layer2_norm(layer2_out)
        layer2_relu = tf.keras.activations.relu(layer2_norm_out)

        layer3_out = self.layer3(layer2_relu)
        layer3_norm_out = self.layer3_norm(layer3_out)
        layer3_relu = tf.keras.activations.relu(layer3_norm_out)

        layer4_out = self.layer4(layer3_relu)
        layer4_norm_out = self.layer4_norm(layer4_out)
        layer4_relu = tf.keras.activations.relu(layer4_norm_out)

        layer5_out = self.layer5(layer4_relu)
        layer5_sigmoid_out = tf.keras.activations.sigmoid(layer5_out)
        layer5_reshape_out = self.layer5_reshape(layer5_sigmoid_out)

        return layer5_reshape_out

    def build_graph(self):
        dummy = tf.keras.Input(shape=self.custom_input_shape, name = "Decoder Input")
        dec_model = tf.keras.Model(inputs=[dummy], outputs=self.call(dummy))
        dec_model._name  = "Decoder"
        return dec_model

    def summary(self):
        dec_model = self.build_graph()
        return dec_model.summary()

class AutoEncoder(tf.keras.Model):
  """Combines the encoder and decoder into an end-to-end model for training."""
  def __init__(self, custom_input_shape=(-1, 224, 224, 3), ae_flavour="vanilla", enc_net="vgg", latent_dim=128):
    super(AutoEncoder, self).__init__(name = "V" + ae_flavour.lower()[1:] + "AutoEncoder")
    self.custom_input_shape = custom_input_shape[1:]
    self._input_shape = self.custom_input_shape
    self.ae_flavour = ae_flavour.lower()
    self.enc_net = enc_net.lower()
    self.latent_dim = latent_dim
    self.encoder = Encoder(custom_input_shape=self.custom_input_shape, ae_flavour=self.ae_flavour, enc_net=self.enc_net, latent_dim=self.latent_dim)
    self.decoder = Decoder(ae_flavour=self.ae_flavour)
    self.bce_loss = 0
    self.kl_loss = 0
    self.total_loss = 0

  @tf.function(experimental_compile=True)
  def compute_KL_loss(self, inputs):
    if self.ae_flavour == "variational":
        z_mean, z_log_var = inputs
        kl_loss = - 0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    elif self.ae_flavour == "vanilla":
        kl_loss = 0
    return kl_loss

  def call(self, inputs, training=False):
    if self.ae_flavour == "variational":
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
    elif self.ae_flavour == "vanilla":
        z = self.encoder(inputs, training=training)
        z_mean, z_log_var = 0, 0

    self.kl_loss = self.compute_KL_loss((z_mean, z_log_var))

    reconstructed = self.decoder(z)
    # Add KL divergence regularization loss.
    return reconstructed

  def build_graph(self):
      dummy = tf.keras.Input(shape=(224, 224, 3), name = "Input")
      ae_model = tf.keras.Model(inputs=[dummy], outputs=self.call(dummy))
      ae_model._name  = "V" + self.ae_flavour.lower()[1:] + "AutoEncoder"
      return ae_model

  def summary(self):
      ae_model = self.build_graph()
      return ae_model.summary()

# ----------------------------------------------Test Model------------------------------------------------------------ #

def get_model_summary(choice = 'ae'):
    if choice == "enc":
        encoder = Encoder()
        print(encoder.summary())
    elif choice == "dec":
        decoder = Decoder()
        print(decoder.summary())
    elif choice == "ae":
        autoencoder_model = AutoEncoder()
        print(autoencoder_model.summary())
    else:
        print("Wrong Choice!")


choice  = 'ae'
# get_model_summary(choice)

# ----------------------------------------------Define Loss, Optimizer and compute metrics function------------------- #

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()
opt = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

@tf.function(experimental_compile=True)
def compute_train_metrics(inputs):
    '''
    Compute training metrics for custom training loop.\n
    :param x: input to model\n
    :param y: output from model\n
    :return: training metrics i.e loss
    '''
    x, y, mode = inputs
    # Open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation.
    with tf.GradientTape() as tape:
        # Run the forward pass of the layer. The operations that the layer applies to its inputs are going to be recorded on the GradientTape.
        # Logits for this minibatch
        reconstructed = autoencoder_model(x, training=True)
        # Compute the loss value for this minibatch.
        bce_loss = loss_fn(y, reconstructed)
        bce_loss = tf.reduce_mean(bce_loss)
        if "variational" in autoencoder_model.name.lower():
            kl_loss = autoencoder_model.kl_loss
        else:
            kl_loss = 0
        total_loss = tf.reduce_mean(bce_loss + kl_loss)
        scaled_loss = opt.get_scaled_loss(total_loss)

        if mode == "train":
            # Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect to the loss.
            scaled_gradients = tape.gradient(scaled_loss, autoencoder_model.trainable_variables)
            grads = opt.get_unscaled_gradients(scaled_gradients)
            # Run one step of gradient descent by updating the value of the variables to minimize the loss.
            opt.apply_gradients(zip(grads, autoencoder_model.trainable_variables))
            # self.opt.apply_gradients(zip(scaled_gradients, self.trainable_variables))

    return [total_loss, bce_loss, kl_loss], reconstructed

# ----------------------------------------------Run Main Code--------------------------------------------------------- #

restrict_dataset = cfg.restrict_dataset
restriction_size = cfg.restriction_size

if __name__ == "__main__":
    # Get train path lists and train data generator
    if restrict_dataset:
        train_DataLoader = data.DataLoader(TAXONOMY_FILE_PATH, RENDERING_PATH, VOXEL_PATH, "train", batch_size, restrict=restrict_dataset, restriction_size=restriction_size)
    else:
        train_DataLoader = data.DataLoader(TAXONOMY_FILE_PATH, RENDERING_PATH, VOXEL_PATH, "train", batch_size)
    train_path_list = train_DataLoader.path_list

    # Get validation path lists and validation data generator
    if restrict_dataset:
        val_DataLoader = data.DataLoader(TAXONOMY_FILE_PATH, RENDERING_PATH, VOXEL_PATH, "val", batch_size, restrict=restrict_dataset, restriction_size=restriction_size)
    else:
        val_DataLoader = data.DataLoader(TAXONOMY_FILE_PATH, RENDERING_PATH, VOXEL_PATH, "val", batch_size)
    val_path_list = val_DataLoader.path_list
    if not os.path.isdir(checkpoint_path):
        logger.info("No model save directory found, so creating directory at -> {0}".format(checkpoint_path))
        os.mkdir(checkpoint_path)
    else:
        logger.info("Found model save directory at -> {0}".format(checkpoint_path))

    saved_model_files = os.listdir(checkpoint_path)
    saved_model_files = [file for file in saved_model_files if autoencoder_flavour.lower() in file]
    saved_model_files = utils.model_sort(saved_model_files)
    if len(saved_model_files) == 0:
        resume_epoch = 0
        autoencoder_model = AutoEncoder(custom_input_shape=tuple([-1] + list(input_shape)), ae_flavour=autoencoder_flavour, enc_net=encoder_cnn)
        logger.info("Starting Training phase")
    else:
        latest_model = os.path.join(checkpoint_path, saved_model_files[-1])
        autoencoder_model = tf.keras.models.load_model(latest_model, compile=False)
        resume_epoch = int(latest_model.split("_")[-1].split("-")[0])
        logger.info("Resuming Training on Epoch -> {0}".format(resume_epoch + 1))
        logger.info("Loading Model from -> {0}".format(latest_model))

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
    for epoch in range(epochs):
        print("\nepoch {}/{}".format(resume_epoch+ epoch + 1, end_epoch))
        learning_rate = learning_rate_fn(epoch)
        progBar = tf.keras.utils.Progbar(num_training_samples, stateful_metrics=['loss_fn'], verbose=1)
        iou_dict = dict()

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train, tax_id) in enumerate(train_DataLoader.data_gen(train_path_list)):
            train_loss, logits = compute_train_metrics((x_batch_train, y_batch_train, "train"))
            train_total_loss = train_loss[0].numpy()
            train_bce_loss = train_loss[1].numpy()
            train_kl_loss = train_loss[2].numpy()
            values = [('total loss', train_total_loss), ('reconstruction loss', train_bce_loss), ('regularization loss', train_kl_loss)]
            progBar.add(batch_size, values)

            iou = metr.calc_iou_loss(y_batch_train, logits)
            iou_dict = metr.iou_dict_update(tax_id, iou_dict, iou)
            mean_iou_train = metr.calc_mean_iou(iou_dict, mean_iou_train)

        allClass_mean_iou = sum(mean_iou_train.values()) / len(mean_iou_train)

        logger.info("Training IoU -> {0}".format(mean_iou_train))
        logger.info("Overall mean Training IoU -> {0}".format(allClass_mean_iou))

        utils.record_iou_data(1, epoch + 1, mean_iou_train)

        with train_summary_writer.as_default():
            tf.summary.scalar('train_loss', train_total_loss, step=epoch)
            tf.summary.scalar('overall_train_iou', allClass_mean_iou, step=epoch)

        logger.info("Validation phase running now for Epoch - {0}".format(epoch + 1))
        for step, (x_batch_val, y_batch_val, tax_id) in tqdm(enumerate(val_DataLoader.data_gen(val_path_list)), total=num_validation_steps):
            val_loss, logits = compute_train_metrics((x_batch_val, y_batch_val, "val"))
            val_total_loss = val_loss[0].numpy()
            val_bce_loss = val_loss[1].numpy()
            val_kl_loss = val_loss[2].numpy()
            iou = metr.calc_iou_loss(y_batch_val, logits)
            # IoU dict update moved to iou_dict_update function
            iou_dict = metr.iou_dict_update(tax_id, iou_dict, iou)
            mean_iou_val = metr.calc_mean_iou(iou_dict, mean_iou_val)

        allClass_mean_iou = sum(mean_iou_val.values()) / len(mean_iou_val)

        with val_summary_writer.as_default():
            tf.summary.scalar('val_loss', val_total_loss, step=epoch)
            tf.summary.scalar('overall_val_iou', allClass_mean_iou, step=epoch)

        logger.info("Validation IoU -> {0}".format(mean_iou_val))
        logger.info("Overall mean Validation IoU -> {0}".format(allClass_mean_iou))

        # Save validation IoU values in CSV file
        utils.record_iou_data(2, epoch + 1, mean_iou_val)

        # Save Model During Training
        if (epoch + 1) % model_save_frequency == 0:
            model_save_file = autoencoder_flavour + '_autoencoder_{0}_model_epoch_{1}-h5'.format(encoder_cnn, resume_epoch + epoch + 1)
            model_save_file_path = os.path.join(checkpoint_path, model_save_file)
            autoencoder_model.save(model_save_file_path, save_format="tf")
            logger.info("Saving {0} Autoencoder Model at {1}".format("V" + autoencoder_flavour.lower()[1:], model_save_file_path))

    logger.info("End of program execution")
