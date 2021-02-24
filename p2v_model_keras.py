import tensorflow as tf
import json
import os
import glob
import sys
import numpy as np
import rgba2rgb as rgba
import binvox_rw
from PIL import Image
import cv2
from tensorflow.keras.applications.vgg16 import VGG16
from tqdm.keras import TqdmCallback
from learning_rate import lr_scheduler

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# tf.debugging.set_log_device_placement(True)

TAXONOMY_FILE_PATH    = 'C:\\Users\\bidnu\\Documents\\Suraj_Docs\\3D_Project\\ShapeNet_P2V\\ShapeNet.json'
RENDERING_PATH        = 'C:\\Users\\bidnu\\Documents\\Suraj_Docs\\3D_Project\\ShapeNet_P2V\\ShapeNetRendering\\{}\\{}\\rendering'
VOXEL_PATH            = 'C:\\Users\\bidnu\\Documents\\Suraj_Docs\\3D_Project\\ShapeNet_P2V\\ShapeNetVox32\\{}\\{}\\model.binvox'

# global test_iou
# global mean_iou

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
            path_list.append([img_path, target_path, taxonomy_dict[i]["taxonomy_id"]])
      
  return path_list

train_path_list = get_xy_paths(taxonomy_dict = taxonomy_dict, mode = 'train')
# print(sys.getsizeof(train_path_list), "bytes")
# print(len(train_path_list))
# train_path_list = np.asarray(train_path_list)
# print("------------------------------------------------------------------------")

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
  
  # IoU per taxonomy
  # test_iou = dict()
  # for i in sample:
  #   if i not in test_iou:
  #     test_iou[i] = {'n_samples': 0, 'iou': []}
  #   test_iou[i]['n_samples'] += 1
  #   test_iou[i]['iou'].append(iou.numpy().tolist())
  
  # # Mean IoU
  # mean_iou = []
  # for taxonomy_id in test_iou:
  #   test_iou[taxonomy_id]['iou'] = np.mean(test_iou[taxonomy_id]['iou'], axis=0)
    # mean_iou.append(test_iou[taxonomy_id]['iou'] * test_iou[taxonomy_id]['n_samples'])

  # TODO: n_samples not defined. Check
  # mean_iou = np.sum(mean_iou, axis=0) / n_samples
   
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
      global img
      img = []
      global target 
      target = []
      global sample
      sample = []
      for file in file_chunk:
        # img_path = file[0].strip('\n')
        # voxel_path = file[1].strip('\n')
        img_path = file[0]
        voxel_path = file[1]
        class_name = file[2]

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
        sample.append(class_name)

    img = np.asarray(img).reshape(-1,224,224,3).astype(np.float32)
    target = np.asarray(target).reshape(-1,32,32,32).astype(np.float32)
    sample = np.asarray(sample).reshape(-1,1).astype(str)

    # print(img.nbytes)
    # print(target.nbytes)
    # print(img.itemsize)
    # print(target.itemsize)
    yield img, target, sample
    # del img
    # del target
    # del sample
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

if __name__ == '__main__':
  input_shape = (224, 224, 3)
  input = tf.keras.Input(shape = input_shape,
                        name = "input_layer")
  encoder_model = tf.keras.Model(input, encoder(input), name = "encoder")
  # encoder_model.summary()
  batch_size = 1
  # print(keras_model_memory_usage_in_bytes(encoder_model, batch_size))
  print("-------------------------")

  decoder_input = tf.keras.Input(shape=(2, 2, 2, 256),
                                name = "decoder_input")

  decoder_model = tf.keras.Model(decoder_input, decoder(decoder_input), name = "decoder")
  # decoder_model.summary()
  # print(keras_model_memory_usage_in_bytes(decoder_model, batch_size))
  print("-------------------------")

  # Autoencoder
  encoder_output = encoder_model(input)
  # the encoder output should be reshaped to (-1,2,2,2,256) to be fed into decoder
  decoder_input = tf.keras.layers.Reshape((2,2,2,256))(encoder_output)

  autoencoder_model = tf.keras.Model(input, decoder_model(decoder_input), name ='autoencoder')
  autoencoder_model.summary()

  # Loss function
  loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits = True)

  # Metric
  # metric = tf.keras.metrics.BinaryCrossentropy(name="binary_crossentropy")
  # need to add intersection over union here as metric

  #  optimizer
  # TODO: Add learning rate scheduler to custom training loop
  opt = tf.keras.optimizers.Adam()

#   # compile_model
#   autoencoder_model.compile(optimizer = opt, loss = loss_fn, metrics = [calc_iou_loss])
  
  # TODO: third output shape not defined properly. Check. Related to IoU calculation
  train_dataset = tf.data.Dataset.from_generator(tf_data_generator,args= [train_path_list, batch_size],
                                          output_types = (tf.float32, tf.float32, tf.string),
                                          output_shapes = ((None, 224, 224, 3),(None, 32, 32, 32),(None, 1)))
  train_dataset.prefetch(tf.data.AUTOTUNE)
#   # print(len(train_path_list))
#   steps_per_epoch = len(train_path_list) // batch_size

# #   # fit model
# #   autoencoder_model.fit(dataset, epochs = 5, verbose = 0,
# #                         steps_per_epoch = steps_per_epoch,
# #                         workers = 4,
# #                         use_multiprocessing = True,
# #                         callbacks=[TqdmCallback(verbose=2)])

#   # Callbacks
#   callbacks = []
#   # Checkpoint
#   filepath = "saved-model-{epoch:02d}-{val_acc:.2f}.hdf5"
#   checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, mode='max', period = 1)
#   callbacks.append(checkpoint)

#   # LR Scheduler
#   callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1))

#   # fit model
#   autoencoder_model.fit(dataset, 
#   						          epochs = 5, 
#   						          verbose = 0,
#                         steps_per_epoch = steps_per_epoch,
#                         workers = 4,
#                         use_multiprocessing = True,
#                         callbacks=callbacks)
  
  # # TODO: Do we need these as global anymore?
  # global test_iou
  # global mean_iou

  # TODO: add checkpoint to custom training loop
  num_training_samples = len(train_path_list)
  epochs = 2
  for epoch in range(1, epochs+1):
    print("\nepoch {}/{}".format(epoch + 1,epochs))

    progBar = tf.keras.utils.Progbar(num_training_samples, stateful_metrics=['loss_fn'])

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train, tax_id) in enumerate(train_dataset):
      tax_id = tax_id.numpy().tolist()

      # Open a GradientTape to record the operations run
      # during the forward pass, which enables auto-differentiation.
      with tf.GradientTape() as tape:

        # Run the forward pass of the layer.
        # The operations that the layer applies
        # to its inputs are going to be recorded
        # on the GradientTape.
        logits = autoencoder_model(x_batch_train, training = True)  # Logits for this minibatch

        # Compute the loss value for this minibatch.
        loss_value = loss_fn(y_batch_train, logits)

      # Use the gradient tape to automatically retrieve
      # the gradients of the trainable variables with respect to the loss.
      grads = tape.gradient(loss_value, autoencoder_model.trainable_weights)

      # Run one step of gradient descent by updating
      # the value of the variables to minimize the loss.
      opt.apply_gradients(zip(grads, autoencoder_model.trainable_weights))

      iou = calc_iou_loss(y_batch_train, logits)

      tax_id = [item for items in tax_id for item in items]

      # TODO: for some reason test_iou has to be defined here itself otherwise getting an error. Check
      test_iou = dict()
      for i in tax_id:
        if i not in test_iou:
          test_iou[i] = {'n_samples': 0, 'iou': []}
        test_iou[i]['n_samples'] += 1
        test_iou[i]['iou'].append(iou.numpy().tolist())
      
      # Mean IoU
      mean_iou = []
      for taxonomy_id in test_iou:
        test_iou[taxonomy_id]['iou'] = np.mean(test_iou[taxonomy_id]['iou'], axis=0)
      
      mean_class_iou = []
      for taxonomy_id in test_iou:
        mean_class_iou.append(test_iou[taxonomy_id]['iou'])
      # mean_class_iou = json.loads(mean_class_iou) #JSONify the mean iou list containing mean iou for each class
      mean_class_iou = json.dumps(mean_class_iou) #JSONify the mean iou list containing mean iou for each class

      # TODO: not able to access the test_iou or mean_iou variables even though they are global. Check
      values=[('train_loss',loss_value),('IoU_metric',iou)]

      # the if loop is just for dubugging purposes
      # if step*batch_size == 500:
      #   break
      
      # FOR TESTING -> Uncomment next if clause to check if the model can be saved and loaded
      if step % 500 == 0:
        print("[STEP = %d] --> [TRAINING LOSS = %.4f] --> [TRAINING IOU = %s]" % (step, float(loss_value), mean_class_iou))
        file_path = 'ae_model_step_{}.h5'.format(step)
        tf.keras.models.save_model(model = autoencoder_model, filepath = file_path, overwrite = False, include_optimizer = True)

      progBar.update(step*batch_size, values=values)

    progBar.update(num_training_samples, values=values, finalize=True)

    print(test_iou)
    
    # FOR TRAINING -> Uncomment during actual training
#   if epoch % 10 == 0:
#   	print("[EPOCH = %d] --> [TRAINING LOSS = %.4f] --> [TRAINING IOU = %s]" % (epoch, float(loss_value), mean_class_iou))
#   	file_path = 'ae_model_epoch_{}.h5'.format(epoch)
#   	tf.keras.models.save_model(model = autoencoder_model, filepath = file_path, overwrite = False, include_optimizer = True)
