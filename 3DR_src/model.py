# ----------------------------------------------Import required Modules----------------------------------------------- #

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16

# ----------------------------------------------Define Model---------------------------------------------------------- #

# Build complete autoencoder model
def build_autoencoder(input_shape = (224, 224, 3), describe = False):
    '''
    Build Autoencoder Model.\n
    :param input_shape: Input Shape passed to Autoencoder Model (224,224,3) (default)\n
    :return: Autoencoder Model
    '''
    def encoder(inp, input_shape=(224,224,3)):
        '''
        Build Encoder Model.\n
        :param inp: Input to Autoencoder Model\n
        :param input_shape: Input Shape passed to Autoencoder Model (224,224,3) (default)\n
        :return: Encoder Model
        '''
        vgg = VGG16(include_top = False,
                    weights = "imagenet",
                    input_shape = input_shape,
                    pooling = "none")

        vgg.trainable = False

        part_vgg = tf.keras.models.Model(inputs = vgg.input,
                                        outputs = vgg.get_layer(name="block4_conv2").output,
                                        name = "part_vgg")

        # https://keras.io/guides/transfer_learning/
        x = part_vgg(inputs = inp, training=False)

        layer10 = tf.keras.layers.Conv2D(filters = 512,
                                         kernel_size = 1,
                                         name = "conv10")(x) # for pix2vox-A(large), kernel_size is 3
        layer10_norm = tf.keras.layers.BatchNormalization(name="layer10_norm")(layer10)
        layer10_elu = tf.keras.activations.elu(layer10_norm,
                                               name="layer10_elu")

        layer11 = tf.keras.layers.Conv2D(filters = 256,
                                         kernel_size = 3,
                                         name = "conv11")(layer10_elu) # for pix2vox-A(large), filters is 512
        layer11_norm = tf.keras.layers.BatchNormalization(name="layer11_norm")(layer11)
        layer11_elu = tf.keras.activations.elu(layer11_norm,
                                               name="layer11_elu")
        layer11_pool = tf.keras.layers.MaxPooling2D(pool_size = (4,4),
                                                    name="layer11_pool")(layer11_elu) # for pix2vox-A(large), kernel size is 3

        layer12 = tf.keras.layers.Conv2D(filters = 128,
                                         kernel_size = 3,
                                         name = "conv12")(layer11_pool) # for pix2vox-A(large), filters is 256, kernel_size is 1
        layer12_norm = tf.keras.layers.BatchNormalization(name="layer12_norm")(layer12)
        layer12_elu = tf.keras.activations.elu(layer12_norm,
                                               name="layer12_elu")

        return layer12_elu

    def decoder(inp):
        '''
        Build Decoder Model.\n
        :param inp: Reshaped Output of Encoder Model\n
        :return: Decoder Model
        '''
        layer1 = tf.keras.layers.Convolution3DTranspose(filters=128,
                                                        kernel_size=4,
                                                        strides=(2,2,2),
                                                        padding="same",
                                                        use_bias=False,
                                                        name="Conv3D_1")(inp)
        layer1_norm = tf.keras.layers.BatchNormalization(name="layer1_norm")(layer1)
        layer1_relu = tf.keras.activations.relu(layer1_norm,
                                                name="layer1_relu")

        layer2 = tf.keras.layers.Convolution3DTranspose(filters=64,
                                                        kernel_size=4,
                                                        strides=(2,2,2),
                                                        padding="same",
                                                        use_bias=False,
                                                        name="Conv3D_2")(layer1_relu)
        layer2_norm = tf.keras.layers.BatchNormalization(name="layer2_norm")(layer2)
        layer2_relu = tf.keras.activations.relu(layer2_norm,
                                                name="layer2_relu")

        layer3 = tf.keras.layers.Convolution3DTranspose(filters=32,
                                                        kernel_size=4,
                                                        strides=(2,2,2),
                                                        padding="same",
                                                        use_bias=False,
                                                        name="Conv3D_3")(layer2_relu)
        layer3_norm = tf.keras.layers.BatchNormalization(name="layer3_norm")(layer3)
        layer3_relu = tf.keras.activations.relu(layer3_norm,
                                                name="layer3_relu")

        layer4 = tf.keras.layers.Convolution3DTranspose(filters=8,
                                                        kernel_size=4,
                                                        strides=(2,2,2),
                                                        padding="same",
                                                        use_bias=False,
                                                        name="Conv3D_4")(layer3_relu)
        layer4_norm = tf.keras.layers.BatchNormalization(name="layer4_norm")(layer4)
        layer4_relu = tf.keras.activations.relu(layer4_norm,
                                                name="layer4_relu")

        layer5 = tf.keras.layers.Convolution3DTranspose(filters=1,
                                                        kernel_size=1,
                                                        use_bias=False,
                                                        name="Conv3D_5")(layer4_relu)
        layer5_sigmoid = tf.keras.activations.sigmoid(layer5,
                                                    name="layer5_sigmoid")

        # TODO: check this statement
        layer5_sigmoid = tf.keras.layers.Reshape((32,32,32))(layer5_sigmoid)

        return layer5_sigmoid

    # Input
    input = tf.keras.Input(shape = input_shape, name = "input_layer")

    # Encoder Model
    encoder_model = tf.keras.Model(input, encoder(input), name = "encoder")
    if describe:
        print("\nEncoder Model Summary:\n")
        encoder_model.summary()

    # Decoder Input Reshaped from Encoder Output
    decoder_input = tf.keras.Input(shape=(2, 2, 2, 256),
                                   name = "decoder_input")

    # Decoder Model
    decoder_model = tf.keras.Model(decoder_input, decoder(decoder_input), name = "decoder")
    if describe:
        print("\nDecoder Model Summary:\n")
        decoder_model.summary()

    # Autoencoder Model
    encoder_output = encoder_model(input)
    # the encoder output should be reshaped to (-1,2,2,2,256) to be fed into decoder
    decoder_input = tf.keras.layers.Reshape((2,2,2,256))(encoder_output)

    autoencoder_model = tf.keras.Model(input, decoder_model(decoder_input), name ='autoencoder')
    if describe:
        print("\nAutoencoder Model Summary:\n")
        autoencoder_model.summary()

    return autoencoder_model