import tensorflow as tf
import json
import os
import glob
import sys
import numpy as np
import rgba
import binvox_rw
from PIL import Image
import cv2
from tensorflow.keras.applications.vgg16 import VGG16
from tqdm.keras import TqdmCallback
from learning_rate import lr_scheduler

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# tf.debugging.set_log_device_placement(True)

TAXONOMY_FILE_PATH    = 'C:\\Users\\bidnu\\Documents\\Suraj_Docs\\Interpolation_2D3D_Project\\Codes\\Pix2Vox\\core\\datasets\\ShapeNet.json'
RENDERING_PATH        = 'C:\\Users\\bidnu\\Documents\\Suraj_Docs\\Interpolation_2D3D_Project\\Codes\\ShapeNet_P2V\\ShapeNetRendering\\{}\\{}\\rendering'
VOXEL_PATH            = 'C:\\Users\\bidnu\\Documents\\Suraj_Docs\\Interpolation_2D3D_Project\\Codes\\ShapeNet_P2V\\ShapeNetVox32\\{}\\{}\\model.binvox'

with open(TAXONOMY_FILE_PATH, encoding='utf-8') as file:
  taxonomy_dict = json.loads(file.read())

def get_xy_paths(taxonomy_dict, mode = 'train'):
  path_list = []
  temp = []
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
            path_list.append([img_path, target_path])
  return path_list

# print(sum(temp))
# for i in train_path_list[20:30]:
#   print(i)
train_path_list = get_xy_paths(taxonomy_dict = taxonomy_dict, mode = 'train')
print(sys.getsizeof(train_path_list), "bytes")
print(len(train_path_list))
train_path_list = np.asarray(train_path_list)
print("------------------------------------------------------------------------")

#Implementation 1 (correct)
def calc_iou_loss(y_true, y_pred):
  y_true = tf.convert_to_tensor(y_true)
  y_pred = tf.convert_to_tensor(y_pred)

  _volume = tf.cast(tf.math.greater_equal(y_pred, 0.3), dtype = tf.float32)
  # print(_volume)

  a = tf.math.multiply(_volume, y_true)
  b = tf.math.reduce_sum(a)
  intersection = tf.cast(b, dtype = tf.float32)

  c = tf.math.add(_volume,y_true)
  d = tf.cast(tf.math.greater_equal(c, 1), dtype = tf.float32)
  # print(d)
  e = tf.math.reduce_sum(d)
  union = tf.cast(e, dtype = tf.float32)

  iou = (intersection / union)
    
  return iou

# y_true = np.random.randint(0,2,size=(32, 32, 32)).astype(np.float32)
# y_pred = np.random.random(size=(32,32,32)).astype(np.float32)

# print("iou - {}".format(calc_iou_loss(y_true, y_pred)))

def tf_data_generator(file_list, batch_size=16):
  i = 0
  while(1):
    if i*batch_size >= len(file_list):
      i = 0
      np.random.shuffle(file_list)
    else:
      file_chunk = file_list[i*batch_size:(i+1)*batch_size]
      img = []
      target = []
      for file in file_chunk:
        # img_path = file[0].strip('\n')
        # voxel_path = file[1].strip('\n')
        img_path = file[0]
        voxel_path = file[1]

        rgba_in = Image.open(img_path)
        rgba_in.load()
        background = Image.new("RGB", rgba_in.size, (255, 255, 255))
        background.paste(rgba_in, mask=rgba_in.split()[3]) # 3 is the alpha channel
        rendering_image = cv2.resize(np.array(background).astype(np.float32), (224,224)) / 255.

        with open(voxel_path, 'rb') as f:
          volume = binvox_rw.read_as_3d_array(f)
          volume = volume.data.astype(np.float32)
        # volume = np.random.random(size=(4,4,128))

        img.append(rendering_image)
        target.append(volume)

    img = np.asarray(img).reshape(-1,224,224,3).astype(np.float32)
    target = np.asarray(target).reshape(-1,32,32,32).astype(np.float32)

    # print(img.nbytes)
    # print(target.nbytes)
    # print(img.itemsize)
    # print(target.itemsize)
    yield img, target
    i = i + 1

# x,y = next(tf_data_generator(train_path_list))
# print(x.shape)
# print(y.shape)

# def fake_decoder_generator(batch_size=16):
#   while(1):
#     encoder_output = np.random.random(size=(4,4,128))
#     encoder_output = np.repeat(encoder_output[np.newaxis,:,:,:],batch_size,axis=0)
#     encoder_output = np.reshape(encoder_output, (-1,4,4,128))

#     decoder_output = np.random.randint(0,2,size=(32,32,32))
#     decoder_output = np.repeat(decoder_output[np.newaxis,:,:,:],batch_size,axis=0)
#     decoder_output = np.reshape(decoder_output, (-1,32,32,32))

#     yield encoder_output, decoder_output

# x,y = next(fake_decoder_generator(8))
# print(x.shape)
# print(y.shape)

# batch_size = 2
# dataset = tf.data.Dataset.from_generator(tf_data_generator,args= [train_path_list, batch_size],
#                                         output_types = (tf.float32, tf.float32),
#                                         output_shapes = ((None,224,224,3),(None,32,32,32)))

# batch_size = 8
# dataset = tf.data.Dataset.from_generator(fake_decoder_generator,args= [batch_size],
#                                         output_types = (tf.float32, tf.int8),
#                                         output_shapes = ((None,4,4,128),(None,32,32,32)))

# print(len(train_path_list))
# steps_per_epoch = len(train_path_list) // batch_size

def encoder(inp, input_shape=(224,224,3)):
  vgg = VGG16(include_top = False,
              weights = "imagenet",
              input_shape = input_shape,
              pooling = "none")
  
  vgg.trainable = False
  
  part_vgg = tf.keras.models.Model(inputs = vgg.input,
                                  outputs = vgg.get_layer(name="block4_conv2").output,
                                  name = "part_vgg")
  
  # https://keras.io/guides/transfer_learning/
  x = part_vgg(inputs = inp,
              training=False)

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
  # inp = tf.keras.layers.Reshape((2,2,2,256))(inp)
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
  
  return layer5_sigmoid

if __name__ == '__main__':
  input_shape = (224,224,3)
  input = tf.keras.Input(shape = input_shape,
                          name = "input_layer")
  encoder_model = tf.keras.Model(input, encoder(input), name="encoder")
  # encoder_model.summary()
  batch_size = 2
  # print(keras_model_memory_usage_in_bytes(encoder_model, batch_size))
  print("-------------------------")

  decoder_input = tf.keras.Input(shape=(2,2,2,256),
                        name="decoder_input")

  decoder_model = tf.keras.Model(decoder_input, decoder(decoder_input), name="decoder")
  #decoder_model.summary()
  # print(keras_model_memory_usage_in_bytes(decoder_model, batch_size))
  print("-------------------------")

  # Autoencoder
  encoder_output = encoder_model(input)
  # the encoder output should be reshaped to (-1,2,2,2,256) to be fed into decoder
  decoder_input = tf.keras.layers.Reshape((2,2,2,256))(encoder_output)

  autoencoder_model = tf.keras.Model(input, decoder_model(decoder_input), name ='autoencoder')
  autoencoder_model.summary()

  # Loss function
  loss_fn = tf.keras.losses.BinaryCrossentropy()

  # Metric
  # metric = tf.keras.metrics.BinaryCrossentropy(name="binary_crossentropy")
  # need to add intersection over union here as metric

  #  optimizer
  opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

  # compile_model
  autoencoder_model.compile(optimizer = opt, loss = loss_fn, metrics = [calc_iou_loss])
  
  dataset = tf.data.Dataset.from_generator(tf_data_generator,args= [train_path_list, batch_size],
                                          output_types = (tf.float32, tf.float32),
                                          output_shapes = ((None,224,224,3),(None,32,32,32)))

  # print(len(train_path_list))
  steps_per_epoch = len(train_path_list) // batch_size

#   # fit model
#   autoencoder_model.fit(dataset, epochs = 5, verbose = 0,
#                         steps_per_epoch = steps_per_epoch,
#                         workers = 4,
#                         use_multiprocessing = True,
#                         callbacks=[TqdmCallback(verbose=2)])

  # Callbacks
  callbacks = []
  # Checkpoint
  filepath = "saved-model-{epoch:02d}-{val_acc:.2f}.hdf5"
  checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, mode='max', period = 1)
  callbacks.append(checkpoint)

  # LR Scheduler
  callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1))

  # fit model
  autoencoder_model.fit(dataset, 
  						          epochs = 5, 
  						          verbose = 0,
                        steps_per_epoch = steps_per_epoch,
                        workers = 4,
                        use_multiprocessing = True,
                        callbacks=callbacks)
