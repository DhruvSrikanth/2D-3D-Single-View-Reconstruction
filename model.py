# ----------------------------------------------Import required Modules----------------------------------------------- #

import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121
from tensorflow.keras import layers

# ----------------------------------------------Define Model---------------------------------------------------------- #

class Sampling(layers.Layer):
  """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class Encoder(layers.Layer):
  """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""
  def __init__(self, inp, enc_net = "vgg", input_shape=(224,224,3), latent_dim = 64, name='encoder', **kwargs):
    super(Encoder, self).__init__(name=name, **kwargs)
    if enc_net == "vgg":
      self.cnn_model = VGG16(include_top = False,weights = "imagenet", input_shape = input_shape,pooling = "none")
      self.cnn_model.trainable = False
      self.pre_trained = tf.keras.models.Model(inputs = self.cnn_model.input,outputs = self.cnn_model.get_layer(name="block4_conv2").output,name = "vgg")

    elif enc_net == "resnet":
      self.cnn_model = ResNet50(include_top = False,weights = "imagenet",input_shape = input_shape,pooling = "none")
      self.cnn_model.trainable = False
      self.pre_trained = tf.keras.models.Model(inputs = self.cnn_model.input,outputs = self.cnn_model.get_layer(name="conv3_block1_out").output,name = "resnet")

    elif enc_net == "densenet":
      self.cnn_model = DenseNet121(include_top = False,weights = "imagenet",input_shape = input_shape,pooling = "none")
      self.cnn_model.trainable = False
      self.pre_trained = tf.keras.models.Model(inputs = self.cnn_model.input,outputs = self.cnn_model.get_layer(name="pool3_relu").output,name = "densenet")

    self.cnn_block = self.pre_trained(inputs = inp, training=False)
    self.layer10 = layers.Conv2D(filters = 512,kernel_size = 1,name = "conv10")(self.cnn_block) # for pix2vox-A(large), kernel_size is 3
    self.layer10_norm = layers.BatchNormalization(name="layer10_norm")(self.layer10)
    self.layer10_elu = tf.keras.activations.elu(self.layer10_norm,name="layer10_elu")

    self.layer11 = layers.Conv2D(filters = 256,kernel_size = 3,name = "conv11")(self.layer10_elu) # for pix2vox-A(large), filters is 512
    self.layer11_norm = layers.BatchNormalization(name="layer11_norm")(self.layer11)
    self.layer11_elu = tf.keras.activations.elu(self.layer11_norm,name="layer11_elu")
    self.layer11_pool = layers.MaxPooling2D(pool_size = (4,4),name="layer11_pool")(self.layer11_elu) # for pix2vox-A(large), kernel size is 3

    self.layer12 = layers.Conv2D(filters = 128,kernel_size = 3,name = "conv12")(self.layer11_pool) # for pix2vox-A(large), filters is 256, kernel_size is 1
    self.layer12_norm = layers.BatchNormalization(name="layer12_norm")(self.layer12)
    self.layer12_elu = tf.keras.activations.elu(self.layer12_norm,name="layer12_elu")

    self.dense_proj = tf.keras.layers.Flatten()(self.layer12_elu)
    self.dense_mean = layers.Dense(latent_dim, name="z_mean")(self.dense_proj)
    self.dense_log_var = layers.Dense(latent_dim, name="z_log_var")(self.dense_proj)
    self.sampling = Sampling()

  def call(self, inputs):
    x = self.dense_proj(inputs)
    z_mean = self.dense_mean(x)
    z_log_var = self.dense_log_var(x)
    z = self.sampling((z_mean, z_log_var))
    return z_mean, z_log_var, z

class Decoder(layers.Layer):
  """Converts z, the encoded digit vector, back into a readable digit."""
  def __init__(self, inp, name='decoder', **kwargs):
    super(Decoder, self).__init__(name=name, **kwargs)
    self.dense_proj = tf.keras.layers.Dense(2048, activation="relu")(inp)
    self.layer0_reshape = tf.keras.layers.Reshape((2, 2, 2, 256))(self.dense_proj)

    self.layer1 = layers.Convolution3DTranspose(filters=128,kernel_size=4,strides=(2,2,2),padding="same",use_bias=False,name="Conv3D_1")(self.layer0_reshape)
    self.layer1_norm = layers.BatchNormalization(name="layer1_norm")(self.layer1)
    self.layer1_relu = tf.keras.activations.relu(self.layer1_norm,name="layer1_relu")

    self.layer2 = layers.Convolution3DTranspose(filters=64,kernel_size=4,strides=(2,2,2),padding="same",use_bias=False,name="Conv3D_2")(self.layer1_relu)
    self.layer2_norm = layers.BatchNormalization(name="layer2_norm")(self.layer2)
    self.layer2_relu = tf.keras.activations.relu(self.layer2_norm,name="layer2_relu")

    self.layer3 = tf.keras.layers.Convolution3DTranspose(filters=32,kernel_size=4,strides=(2,2,2),padding="same",use_bias=False,name="Conv3D_3")(self.layer2_relu)
    self.layer3_norm = tf.keras.layers.BatchNormalization(name="layer3_norm")(self.layer3)
    self.layer3_relu = tf.keras.activations.relu(self.layer3_norm,name="layer3_relu")

    self.layer4 = tf.keras.layers.Convolution3DTranspose(filters=8,kernel_size=4,strides=(2,2,2),padding="same",use_bias=False,name="Conv3D_4")(self.layer3_relu)
    self.layer4_norm = tf.keras.layers.BatchNormalization(name="layer4_norm")(self.layer4)
    self.layer4_relu = tf.keras.activations.relu(self.layer4_norm,name="layer4_relu")

    self.layer5 = tf.keras.layers.Convolution3DTranspose(filters=1,kernel_size=1,use_bias=False,name="Conv3D_5")(self.layer4_relu)
    self.layer5_sigmoid = tf.keras.activations.sigmoid(self.layer5,name="layer5_sigmoid")

    self.dense_output = tf.keras.layers.Reshape((32,32,32))(self.layer5_sigmoid)

  def call(self, inputs):
    x = self.dense_proj(inputs)
    return self.dense_output(x)

class VariationalAutoEncoder(tf.keras.Model):
  """Combines the encoder and decoder into an end-to-end model for training."""
  def __init__(self, inputs, input_shape = (224, 224, 3), enc_net = "vgg", name="VAE", **kwargs):
    super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
    self.encoder = Encoder(inputs, input_shape=input_shape, enc_net=enc_net)
    self.decoder = Decoder(inputs)

  def call(self, inputs):
    # self._set_inputs(inputs)
    z_mean, z_log_var, z = self.encoder(inputs)
    reconstructed = self.decoder(z)
    # Add KL divergence regularization loss.
    kl_loss = - 0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    self.add_loss(kl_loss)
    return reconstructed

inputs = tf.keras.Input(shape=(224, 224, 3), name="input_layer")
train_model = VariationalAutoEncoder(inputs, input_shape = (224, 224, 3), enc_net = "vgg")
print(train_model.summary())