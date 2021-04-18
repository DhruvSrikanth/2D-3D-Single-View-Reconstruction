# ----------------------------------------------Import required Modules----------------------------------------------- #

import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121

# ----------------------------------------------Define Model---------------------------------------------------------- #

class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class Encoder(tf.keras.Model):

    def __init__(self, custom_input_shape=(224, 224, 3), ae_flavour="vanilla", enc_net="vgg", latent_dim=128):
        super(Encoder, self).__init__(name="Encoder")
        # Variables
        self.custom_input_shape = custom_input_shape
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

class Decoder(tf.keras.Model):

    def __init__(self, ae_flavour="vanilla"):
        super(Decoder, self).__init__(name="Decoder")
        # Variables
        self.ae_flavour = ae_flavour.lower()
        # CNN-Block

        if self.ae_flavour == "variational":
            self.layer0 = tf.keras.layers.Dense(2048, activation="relu")
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
            layer0_reshape_out = self.layer0_reshape(layer0_out)
            layer1_in = layer0_reshape_out

        elif self.ae_flavour == "vanilla":
            layer1_in = inputs

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

class AutoEncoder(tf.keras.Model):
  """Combines the encoder and decoder into an end-to-end model for training."""

  def __init__(self, custom_input_shape=(-1, 224, 224, 3), ae_flavour="vanilla", enc_net="vgg", latent_dim=128):
    super(AutoEncoder, self).__init__(name = "V" + ae_flavour.lower()[1:] + "AutoEncoder")
    self.custom_input_shape = custom_input_shape[1:]
    self.ae_flavour = ae_flavour.lower()
    self.enc_net = enc_net.lower()
    self.latent_dim = latent_dim
    self.encoder = Encoder(custom_input_shape=self.custom_input_shape, ae_flavour=self.ae_flavour, enc_net=self.enc_net, latent_dim=self.latent_dim)
    self.decoder = Decoder(ae_flavour=self.ae_flavour)

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
        kl_loss = self.compute_KL_loss((z_mean, z_log_var))
    elif self.ae_flavour == "vanilla":
        z = self.encoder(inputs, training=training)
        kl_loss = 0

    reconstructed = self.decoder(z)
    # Add KL divergence regularization loss.
    self.add_loss(kl_loss)
    return reconstructed