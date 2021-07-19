# ----------------------------------------------Import required Modules----------------------------------------------- #
import tensorflow as tf
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    Conv2DTranspose, Conv3D, Conv3DTranspose, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.initializers import glorot_uniform


# ----------------------------------------------Define ResNet50Kern Model--------------------------------------------- #

def factored_conv33(X, f, filters, name, s=1):
    # 3x3 conv is represented as 1x3 followed by 3x1 conv
    X = Conv2D(filters, kernel_size=(1, f), strides=(s, 1), kernel_initializer=glorot_uniform(seed=0), padding='same',
               name=name + '_1')(X)
    X = Conv2D(filters, kernel_size=(f, 1), strides=(1, s), kernel_initializer=glorot_uniform(seed=0), padding='same',
               name=name + '_2')(X)
    return X


def identity_block(X, f, filters, stage, block, s=2):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = factored_conv33(X, f, filters=F2, name=conv_name_base + '2b', s=1)
    # X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = factored_conv33(X, f, filters=F2, name=conv_name_base + '2b', s=1)
    # X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def ResNet50Kern(X_input, input_shape=(224, 224, 3)):
    # Define the input as a tensor with shape input_shape
    # X_input = Input(input_shape)

    # zero padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1 (observations mentioned in the end)
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    # X = factored_conv33(X_input, 7, 64, name='conv1a', s=2)
    # X = factored_conv33(X, 3, 64, name='conv1b', s=1)
    # X = factored_conv33(X, 3, 64, name='conv1c', s=1)

    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # # ### START CODE HERE ###

    # # Stage 3 (≈4 lines)
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)

    return X


# Obsevations
# without kernel splitting (3 and 7) total parameters are ~613k
# with complete kernel splitting (7 reduced to 1x3 and 3x1, same with 3) total params are ~597k
# with onlu 3x3 kernel splitting total params are ~527k

# ----------------------------------------------Define Autoencoder Resnet50 Kernel Model---------------------------------------------------------- #

# Build complete autoencoder model
def build_autoencoder(input_shape=(224, 224, 3), describe=False):
    '''
    Build Autoencoder Model.\n
    :param input_shape: Input Shape passed to Autoencoder Model (224,224,3) (default)\n
    :return: Autoencoder Model
    '''

    def conv2D_block(X, f, filters, s, block):

        conv_name_base = 'conv_block_' + str(block) + '_branch_'
        bn_name_base = 'bn_block_' + str(block) + '_branch_'
        mp_name_base = 'mp_block_' + str(block) + '_branch_'
        add_name_base = 'add_block_' + str(block) + '_branch_'

        print('\nEncoder Block {} - '.format(str(block)))

        F1, F2, F3 = filters

        X_shortcut = X

        X = tf.keras.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s),
                                   kernel_initializer=glorot_uniform(seed=0), name=conv_name_base + '1a')(
            X)  # for pix2vox-A(large), kernel_size is 3
        X = tf.keras.layers.BatchNormalization(name=bn_name_base + '1a')(X)
        X = tf.keras.activations.elu(X)
        print('main path (post 1st conv) shape = ', X.shape)

        X = tf.keras.layers.Conv2D(filters=F2, kernel_size=(1, f), strides=(1, 1), padding='valid',
                                   kernel_initializer=glorot_uniform(seed=0), name=conv_name_base + '2a_1')(
            X)  # for pix2vox-A(large), filters is 512
        X = tf.keras.layers.Conv2D(filters=F2, kernel_size=(f, 1), strides=(1, 1), padding='valid',
                                   kernel_initializer=glorot_uniform(seed=0), name=conv_name_base + '2a_2')(
            X)  # for pix2vox-A(large), filters is 512
        X = tf.keras.layers.BatchNormalization(name=bn_name_base + '2a')(X)
        X = tf.keras.activations.elu(X)
        X = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), name=mp_name_base + '2a')(
            X)  # for pix2vox-A(large), kernel size is 3
        print('main path (post 2nd conv) shape = ', X.shape)

        X = tf.keras.layers.Conv2D(filters=F3, kernel_size=(1, f), strides=(1, 1), padding='valid',
                                   kernel_initializer=glorot_uniform(seed=0), name=conv_name_base + '3a_1')(X)
        X = tf.keras.layers.Conv2D(filters=F3, kernel_size=(f, 1), strides=(1, 1), padding='valid',
                                   kernel_initializer=glorot_uniform(seed=0), name=conv_name_base + '3a_2')(
            X)  # for pix2vox-A(large), filters is 256, kernel_size is 1
        X = tf.keras.layers.BatchNormalization(name=bn_name_base + '3a')(X)
        print('main path (post 3rd conv) shape = ', X.shape)

        X_shortcut = tf.keras.layers.Conv2D(filters=F3, kernel_size=(1, f), strides=(s + 6, s + 6), padding='valid',
                                            kernel_initializer=glorot_uniform(seed=0), name=conv_name_base + '1b_1')(
            X_shortcut)  # for pix2vox-A(large), filters is 256, kernel_size is 1
        X_shortcut = tf.keras.layers.Conv2D(filters=F3, kernel_size=(f, 1), strides=(s + 6, s + 6), padding='valid',
                                            kernel_initializer=glorot_uniform(seed=0), name=conv_name_base + '1b_2')(
            X_shortcut)  # for pix2vox-A(large), filters is 256, kernel_size is 1
        X_shortcut = tf.keras.layers.BatchNormalization(name=bn_name_base + '1b')(X_shortcut)
        print('shortcut  (post 1st conv) shape = ', X_shortcut.shape)

        X = tf.keras.layers.Add(name=add_name_base + '4a')([X, X_shortcut])
        X = tf.keras.activations.elu(X)

        return X

    def pre_train_conv_block(inp, input_shape=input_shape):
        '''
        Build Pre Trained Model.\n
        :param inp: Input to Autoencoder Model\n
        :param input_shape: Input Shape passed to Autoencoder Model (224,224,3) (default)\n
        :return: Pre Trained Model
        '''

        cnn_model = tf.keras.Model(input, ResNet50Kern(input), name="resnet_kr")
        # cnn_model = ResNet50(include_top = False, weights = "imagenet", input_shape = input_shape, pooling = "none")
        cnn_model.trainable = False
        # pre_trained = tf.keras.models.Model(inputs = cnn_model.input, outputs = cnn_model.get_layer(name="conv3_block1_out").output, name = "resnet")
        pre_trained = cnn_model
        # https://keras.io/guides/transfer_learning/
        x = pre_trained(inputs=inp, training=False)
        # print(pre_trained.summary())

        return x

    def encoder(inp, input_shape=input_shape):
        '''
        Build Encoder Model.\n
        :param inp: Input to Autoencoder Model\n
        :param input_shape: Input Shape passed to Autoencoder Model (224,224,3) (default)\n
        :return: Encoder Model
        '''

        print('input shape = ', inp.shape)

        x = pre_train_conv_block(inp, input_shape)
        enc_out = conv2D_block(x, f=3, filters=[512, 256, 128], s=1, block=2)

        return enc_out

    def deconv3D_block(X, f, filters, s, block):

        print('\nDecoder Block {} - '.format(str(block)))
        deconv_name_base = 'deconv_block_' + str(block) + '_branch_'
        bn_name_base = 'bn_block_' + str(block) + '_branch_'
        add_name_base = 'add_block_' + str(block) + '_branch_'

        F1, F2 = filters

        X_shortcut = X

        X = tf.keras.layers.Convolution3DTranspose(filters=F1, kernel_size=(f, f, f), strides=(s, s, s), padding="same",
                                                   kernel_initializer=glorot_uniform(seed=0), use_bias=False,
                                                   name=deconv_name_base + '1a')(X)
        X = tf.keras.layers.BatchNormalization(name=bn_name_base + '1a')(X)
        X = tf.keras.activations.relu(X)
        print('main path (post 1st conv) shape = ', X.shape)

        X = tf.keras.layers.Convolution3DTranspose(filters=F2, kernel_size=(f, f, f), strides=(s, s, s), padding="same",
                                                   kernel_initializer=glorot_uniform(seed=0), use_bias=False,
                                                   name=deconv_name_base + '2a')(X)
        X = tf.keras.layers.BatchNormalization(name=bn_name_base + '2a')(X)
        print('main path (post 2nd conv) shape = ', X.shape)

        X_shortcut = tf.keras.layers.Convolution3DTranspose(filters=F2, kernel_size=(f, f, f),
                                                            strides=(s + 2, s + 2, s + 2), padding='valid',
                                                            kernel_initializer=glorot_uniform(seed=0), use_bias=False,
                                                            name=deconv_name_base + '1b')(X_shortcut)
        X_shortcut = tf.keras.layers.BatchNormalization(name=bn_name_base + '1b')(X_shortcut)
        print('shortcut  (post 1st conv) shape = ', X_shortcut.shape)

        X = tf.keras.layers.Add(name=add_name_base + '3a')([X, X_shortcut])
        X = tf.keras.activations.relu(X)

        return X

    def decoder(inp):
        '''
        Build Decoder Model.\n
        :param inp: Reshaped Output of Encoder Model\n
        :return: Decoder Model
        '''

        deconv_name_base = 'deconv_block_' + str(3) + '_branch_'

        x = deconv3D_block(inp, f=4, filters=[128, 64], s=2, block=1)
        x = deconv3D_block(x, f=4, filters=[32, 8], s=2, block=2)

        x = tf.keras.layers.Convolution3DTranspose(filters=1, kernel_size=1, padding='same',
                                                   kernel_initializer=glorot_uniform(seed=0), use_bias=False,
                                                   name=deconv_name_base + '1a')(x)
        x = tf.keras.activations.sigmoid(x)

        dec_out = tf.keras.layers.Reshape((32, 32, 32))(x)

        print('\noutput shape = ', dec_out.shape)

        return dec_out

    # Input
    input = tf.keras.Input(shape=input_shape, name="input_layer")

    # Encoder Model
    encoder_model = tf.keras.Model(input, encoder(input), name="encoder")
    if describe:
        print("\nEncoder Model Summary:\n")
        encoder_model.summary()

    # Decoder Input Reshaped from Encoder Output
    decoder_input = tf.keras.Input(shape=(2, 2, 2, 256), name="decoder_input")

    # Decoder Model
    decoder_model = tf.keras.Model(decoder_input, decoder(decoder_input), name="decoder")
    if describe:
        print("\nDecoder Model Summary:\n")
        decoder_model.summary()

    # Autoencoder Model
    encoder_output = encoder_model(input)
    # the encoder output should be reshaped to (-1,2,2,2,256) to be fed into decoder
    decoder_input = tf.keras.layers.Reshape((2, 2, 2, 256))(encoder_output)

    autoencoder_model = tf.keras.Model(input, decoder_model(decoder_input), name='autoencoder')
    if describe:
        print("\nAutoencoder Model Summary:\n")
        autoencoder_model.summary()

    return autoencoder_model


# autoencoder_model = build_autoencoder()
# print(autoencoder_model.summary())