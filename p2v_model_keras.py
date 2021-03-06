# ----------------------------------------------Import required Modules----------------------------------------------- #

import os
import sys
import glob
import json
import numpy as np
import datetime

import cv2
from PIL import Image
import rgba2rgb as rgba
import binvox_rw

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16

# ----------------------------------------------Set Environment Variables--------------------------------------------- #

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# tf.debugging.set_log_device_placement(True)

# ----------------------------------------------Set File Paths-------------------------------------------------------- #

TAXONOMY_FILE_PATH    = 'C:\\Users\\bidnu\\Documents\\Suraj_Docs\\3D_Project\\ShapeNet_P2V\\ShapeNet.json'
RENDERING_PATH        = 'C:\\Users\\bidnu\\Documents\\Suraj_Docs\\3D_Project\\ShapeNet_P2V\\ShapeNetRendering\\{}\\{}\\rendering'
VOXEL_PATH            = 'C:\\Users\\bidnu\\Documents\\Suraj_Docs\\3D_Project\\ShapeNet_P2V\\ShapeNetVox32\\{}\\{}\\model.binvox'

# ----------------------------------------------Training Configuration------------------------------------------------ #

input_shape = (224, 224, 3)  # input shape
batch_size = 1  # batch size
epochs = 4  # Number of epochs
model_save_frequency = 2 # Save model every n epochs (specify n)

# ----------------------------------------------Define Dataset Reader and Generator----------------------------------- #

def read_taxonomy_JSON(filepath):
    '''
    Read JSON file containing dataset taxonomy.
    :param filepath: JSON file path
    :return: Un-JSON-ified dictionary
    '''
    with open(filepath, encoding='utf-8') as file:
        taxonomy_dict = json.loads(file.read())
    return taxonomy_dict

def get_xy_paths(taxonomy_dict, mode = 'train'):
    '''
    Get list of file paths for x (images) and y (voxel models).
    :param taxonomy_dict: Dataset Taxonomy Dictionary
    :param mode: Dataset type -> 'train' (default), 'test'
    :return: List containing file path for x and corresponding y
    '''
    path_list = []
    for i in range(len(taxonomy_dict)):
        for sample in taxonomy_dict[i][mode]:
            render_txt = os.path.join(RENDERING_PATH.format(taxonomy_dict[i]["taxonomy_id"], sample), "renderings.txt")
            if not os.path.exists(render_txt):
                continue
            with open(render_txt, 'r') as f:
                while(1):
                    value = next(f,'end')
                    if value == 'end':
                        break
                    else:
                        img_path = os.path.join(RENDERING_PATH.format(taxonomy_dict[i]["taxonomy_id"], sample), value.strip('\n'))
                        target_path = VOXEL_PATH.format(taxonomy_dict[i]["taxonomy_id"], sample)
                        path_list.append([img_path, target_path, taxonomy_dict[i]["taxonomy_id"]])
    return path_list

def tf_data_generator(file_list, batch_size=16):
    '''
    Create generator from file path list.
    :param file_list: List of file paths
    :param batch_size: Batch Size
    :return: Generator object
    '''
    i = 0
    while(1):
        if i*batch_size >= len(file_list):
            i = 0
            np.random.shuffle(file_list)
        else:
            file_chunk = file_list[i*batch_size:(i+1)*batch_size]
            global img
            img = []
            global target
            target = []
            global sample
            sample = []
            for file in file_chunk:
                img_path = file[0]
                voxel_path = file[1]
                class_name = file[2]

                rgba_in = Image.open(img_path)
                # rgba_in.load()
                background = Image.new("RGB", rgba_in.size, (255, 255, 255))
                background.paste(rgba_in, mask=rgba_in.split()[3]) # 3 is the alpha channel
                rendering_image = cv2.resize(np.array(background).astype(np.float32), (224,224)) / 255.

                with open(voxel_path, 'rb') as f:
                    volume = binvox_rw.read_as_3d_array(f)
                    volume = volume.data.astype(np.float32)

                    img.append(rendering_image)
                    target.append(volume)
                    sample.append(class_name)

    img = np.asarray(img).reshape(-1,224,224,3).astype(np.float32)
    target = np.asarray(target).reshape(-1,32,32,32).astype(np.float32)
    sample = np.asarray(sample).reshape(-1,1).astype(str)

    yield img, target, sample
    i = i + 1

# TODO: look at data augmentation becuase there is a class imbalance of images

def tf_data_generator2(file_list):
    '''
    Create generator from file path list.
    :param file_list: List of file paths
    :return: Generator object
    '''
    for img, voxel, tax_id in file_list:
        rgba_in = Image.open(img)
        # rgba_in.load()
        background = Image.new("RGB", rgba_in.size, (255, 255, 255))
        background.paste(rgba_in, mask=rgba_in.split()[3]) # 3 is the alpha channel
        rendering_image = cv2.resize(np.array(background).astype(np.float32), (224,224)) / 255.

        with open(voxel, 'rb') as f:
          volume = binvox_rw.read_as_3d_array(f)
          volume = volume.data.astype(np.float32)

        yield rendering_image, volume, tax_id

# ----------------------------------------------Define Model---------------------------------------------------------- #

# Build complete autoencoder model
def build_autoencoder(input_shape = (224, 224, 3)):
    '''
    Build Autoencoder Model.
    :param input_shape: Input Shape passed to Autoencoder Model (224,224,3) (default)
    :return: Autoencoder Model
    '''
    def encoder(inp, input_shape=(224,224,3)):
        '''
        Build Encoder Model.
        :param inp: Input to Autoencoder Model
        :param input_shape: Input Shape passed to Autoencoder Model (224,224,3) (default)
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
        Build Decoder Model.
        :param inp: Reshaped Output of Encoder Model
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
    # encoder_model.summary()

    # Decoder Input Reshaped from Encoder Output
    decoder_input = tf.keras.Input(shape=(2, 2, 2, 256),
                                   name = "decoder_input")

    # Decoder Model
    decoder_model = tf.keras.Model(decoder_input, decoder(decoder_input), name = "decoder")
    # decoder_model.summary()

    # Autoencoder Model
    encoder_output = encoder_model(input)
    # the encoder output should be reshaped to (-1,2,2,2,256) to be fed into decoder
    decoder_input = tf.keras.layers.Reshape((2,2,2,256))(encoder_output)

    autoencoder_model = tf.keras.Model(input, decoder_model(decoder_input), name ='autoencoder')
    # autoencoder_model.summary()
    # print("-------------------------")

    return autoencoder_model

# ----------------------------------------------Define Optimizer------------------------------------------------------ #

#Implementation 1 (correct)
# TODO: the output of this function for some reason cannot be appended to a list in the training loop
# We need to take care when we pass batch inputs. right now I have added a crude implementation of that but try and make it better

# Calculate IOU loss
def calc_iou_loss(y_true, y_pred):
    '''
    Calculate Intersection Over Union for the given batch
    :param y_true: Target Voxel Output
    :param y_pred: Predicted Voxel Output
    :return: IoU for batch (list)
    '''
    # y_true = tf.convert_to_tensor(y_true)
    # y_pred = tf.convert_to_tensor(y_pred)
    res = []
    bs = y_true.shape[0]
    for i in range(bs):
        # TF implementation
        # _volume = tf.cast(tf.math.greater_equal(y_pred, 0.3), dtype = tf.float32)
        # a = tf.math.multiply(_volume, y_true)
        # b = tf.math.reduce_sum(a)
        # intersection = tf.cast(b, dtype = tf.float32)
        # c = tf.math.add(_volume,y_true)
        # d = tf.cast(tf.math.greater_equal(c, 1), dtype = tf.float32)
        # e = tf.math.reduce_sum(d)
        # union = tf.cast(e, dtype = tf.float32)
        # iou = (intersection / union)

        # Numpy Implementation
        _volume = np.greater_equal(y_pred[i], 0.3).astype(np.float32)
        a = np.multiply(_volume, y_true[i])
        b = np.sum(a)
        intersection = b.astype(np.float32)
        c = np.add(_volume, y_true[i])
        d = np.greater_equal(c, 1).astype(np.float32)
        e = np.sum(d)
        union = e.astype(np.float32)
        iou = np.divide(intersection, union)

        res.append(iou.tolist())
    return res

# Test Values for IOU Loss
# y_true = np.random.randint(0,2,size=(32, 32, 32)).astype(np.float32)
# y_pred = np.random.random(size=(32,32,32)).astype(np.float32)

# ans = calc_iou_loss(y_true, y_pred)
# print(ans)
# print("iou - {}".format(ans))

# TODO: Function needs to be revisited, no return value specifiec
def iou_dict_update(tax_id):
    '''
    Update IOU dictionary for each class.
    :param tax_id: Class ID
    :return: Updated IOU Dictionary
    '''
    for i, j in enumerate(tax_id):
      # print(i)
      if j not in test_iou:

        # TODO: not able to append the iou list to the iou filed of the dictionary below. This is a temporary workaround
        # @dhruv @rishab try fixing this issue

        test_iou[j] = {'n_samples': 0, 'iou': 0.0}

      test_iou[j]['n_samples'] += 1
      test_iou[j]['iou'] += iou[i]

      # # Mean IoU
      # mean_iou = []
      for taxonomy_id in test_iou:
          test_iou[taxonomy_id]['iou'] = test_iou[taxonomy_id]['iou'] / test_iou[taxonomy_id]['n_samples']

# ----------------------------------------------Train Function-------------------------------------------------------- #

# Custom Train Function
def my_train(x,y):
    '''
    Compute training metrics for custom training loop.
    :param x: input to model
    :param y: output from model
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
    taxonomy_dict = read_taxonomy_JSON(TAXONOMY_FILE_PATH)

    # Get path lists
    train_path_list = get_xy_paths(taxonomy_dict=taxonomy_dict, mode='train')
    train_path_list_sample = train_path_list[:100] + train_path_list[-100:]  # just for testing purposes

    val_path_list = get_xy_paths(taxonomy_dict=taxonomy_dict, mode='val')
    val_path_list_sample = val_path_list[:20] + val_path_list[-20:]  # just for testing purposes

    # TODO: third output shape not defined properly. Check. Related to IoU calculation
    # train_dataset = tf.data.Dataset.from_generator(tf_data_generator, args= [train_path_list_sample, batch_size],
    #                                         output_types = (tf.float32, tf.float32, tf.string),
    #                                         output_shapes = ((None, 224, 224, 3),(None, 32, 32, 32),(None, 1)))
    # train_dataset.prefetch(tf.data.AUTOTUNE).cache("cache_file.txt")

    # TODO: since the older generator was running forever, this is a fix but need to check properly again
    train_dataset = tf.data.Dataset.from_generator(tf_data_generator2, args=[train_path_list_sample],
                                          output_types = (tf.float32, tf.float32, tf.string))
    train_dataset = train_dataset.batch(batch_size).shuffle(150).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(tf_data_generator2, args=[val_path_list_sample],
                                          output_types = (tf.float32, tf.float32, tf.string))
    val_dataset = train_dataset.batch(batch_size).shuffle(150).prefetch(tf.data.AUTOTUNE)

    # Load Model and Resume Training, otherwise Start Training
    saved_model_files = glob.glob("*.h5")
    if len(saved_model_files) == 0:
        resume_epoch = 0
        autoencoder_model = build_autoencoder(input_shape)
        print("\nStarting Training")
    else:
        latest_model = saved_model_files[-1]
        autoencoder_model = tf.keras.models.load_model(latest_model, compile=False)
        resume_epoch = int(latest_model.split("_")[-1].split(".")[0])
        print("\nResuming Training On Epoch -> ", resume_epoch)
        print("\nLoading Model From -> ", latest_model)

    # Loss
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Learning Rate Scheduler
    # learning rate becomes 0.01*0.5 after 150 epochs else it is 0.01*1.0
    learning_rate = 0.001
    boundaries = [150]
    values = [learning_rate, learning_rate * 0.5]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

    # Optimizer
    opt = tf.keras.optimizers.Adam()

    # Tensorboard Graph
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # val_log_dir = 'logs/gradient_tape/' + current_time + '/val'
    # val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    mean_iou = list()
    mean_class_iou = list()

    # Training Loop
    num_training_samples = len(train_path_list_sample)
    end_epoch = epochs
    for epoch in range(resume_epoch, end_epoch, 1):
        print("\nepoch {}/{}".format(epoch + 1, end_epoch))

        learning_rate = learning_rate_fn(epoch)

        test_iou = dict()

        progBar = tf.keras.utils.Progbar(num_training_samples, stateful_metrics=['loss_fn'], verbose=1)

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train, tax_id) in enumerate(train_dataset):
            tax_id = tax_id.numpy()
            tax_id = [item.decode("utf-8") for item in tax_id] # byte string (b'hello' to regular string 'hello')

            loss_value, logits = my_train(x_batch_train, y_batch_train)

            iou = calc_iou_loss(y_batch_train, logits)

            # IoU dict update moved to iou_dict_update function
            iou_dict_update(tax_id)

            # mean_class_iou = []
            for taxonomy_id in test_iou:
                mean_iou.append(test_iou[taxonomy_id]['iou'])
            # mean_class_iou = json.loads(mean_class_iou) #JSONify the mean iou list containing mean iou for each class
            # mean_class_iou = json.dumps(mean_class_iou) #JSONify the mean iou list containing mean iou for each class

            # TODO: not able to access the test_iou or mean_iou variables even though they are global. Check
            values=[('train_loss', loss_value)]

            progBar.add(batch_size, values)

            # print(loss_value.numpy())
            # print(test_iou)
            # mean_class_iou = json.dumps(mean_iou) #JSONify the mean iou list containing mean iou for each class

        with train_summary_writer.as_default():
            tf.summary.scalar('train_loss', loss_value, step=epoch)
            tf.summary.scalar('train_iou', iou[0], step=epoch)

        # Save Model During Training
        if (epoch+1) % model_save_frequency == 0:
            model_save_file_path = 'ae_model_epoch_{}.h5'.format(epoch+1)
            print("Saving Autoencoder Model at ", model_save_file_path)
            tf.keras.models.save_model(model=autoencoder_model, filepath=model_save_file_path, overwrite=False, include_optimizer=True)