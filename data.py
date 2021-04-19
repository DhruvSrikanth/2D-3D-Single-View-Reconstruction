# ----------------------------------------------Import required Modules----------------------------------------------- #

import os
import json
import numpy as np
import random

import cv2
from PIL import Image
import binvox_rw

from logger import logger_train, logger_test

# ----------------------------------------------Define Dataset Reader and Generator----------------------------------- #

def read_taxonomy_JSON(filepath):
    '''
    Read JSON file containing dataset taxonomy.\n
    :param filepath: JSON file path\n
    :return: Un-JSON-ified dictionary
    '''
    with open(filepath, encoding='utf-8') as file:
        taxonomy_dict = json.loads(file.read())
    return taxonomy_dict

def get_xy_paths(taxonomy_dict, rendering_path, voxel_path, mode = 'train'):
    '''
    Get list of file paths for x (images) and y (voxel models).\n
    :param taxonomy_dict: Dataset Taxonomy Dictionary\n
    :param mode: Dataset type -> 'train' (default), 'test'\n
    :return: List containing file path for x and corresponding y
    '''
    if mode == "test":
        logger = logger_test
    else:
        logger = logger_train
        
    path_list = []
    logger.info("Starting to read input data files for {0} phase now".format(mode))
    for i in range(len(taxonomy_dict)):
        logger.info("Reading files of Taxonomy ID -> {0}, Class ->{1}".format(taxonomy_dict[i]["taxonomy_id"],
                                                                             taxonomy_dict[i]["taxonomy_name"]))
        for sample in taxonomy_dict[i][mode]:
            render_txt = os.path.join(rendering_path.format(taxonomy_dict[i]["taxonomy_id"], sample), "renderings.txt")
            if not os.path.exists(render_txt):
                logger.warn("Could not find file -> {0}".format(render_txt))
                continue
            with open(render_txt, 'r') as f:
                while(1):
                    value = next(f,'end')
                    if value == 'end':
                        break
                    else:
                        img_path = os.path.join(rendering_path.format(taxonomy_dict[i]["taxonomy_id"], sample),
                                                value.strip('\n'))
                        target_path = voxel_path.format(taxonomy_dict[i]["taxonomy_id"], sample)
                        path_list.append([img_path, target_path, taxonomy_dict[i]["taxonomy_id"]])

    # Shuffle path list
    random.shuffle(path_list) # in-place

    logger.info("Finished reading all the files")
    return path_list

# TODO: look at data augmentation because there is a class imbalance of images (Ask Sir)
# ----> He said check with and without to see if it is required after a performance comparison but thinks it will not be
#       required since classification is not being done in this case

def tf_data_generator(file_list, mode = 'Train'):
    '''
    Create generator from file path list.\n
    :param file_list: List of file paths\n
    :return: Generator object
    '''
    if mode == 'Train' or mode == 'Test':
        for img, voxel, tax_id in file_list:
            rgba_in = Image.open(img)
            # background = Image.new("RGB", rgba_in.size, (255, 255, 255))
            # background.paste(rgba_in, mask=rgba_in.split()[3]) # 3 is the alpha channel
            rendering_image = cv2.resize(np.array(rgba_in).astype(np.float32), (224,224)) / 255.

            with open(voxel, 'rb') as f:
              volume = binvox_rw.read_as_3d_array(f)
              volume = volume.data.astype(np.float32)

            yield rendering_image, volume, tax_id

    elif mode == 'Inference':
        img, voxel = file_list[0], file_list[1]
        rgba_in = Image.open(img)
        rendering_image = cv2.resize(np.array(rgba_in).astype(np.float32), (224, 224)) / 255.

        with open(voxel, 'rb') as f:
            volume = binvox_rw.read_as_3d_array(f)
        volume = volume.data.astype(np.float32)

        yield rendering_image, volume

def data_gen(file_list, batch_size=1):
  '''
  Create generator from file path list.\n
  :param file_list: List of file paths\n
  :param batch_size: batch_size\n
  :return: Generator object
  '''
  # Shuffle path list
  random.shuffle(file_list) # in-place

  l = len(file_list)
  random.shuffle(file_list)

  for idx in range(0,l,batch_size):
      img, vox, tax_id = [],[],[]
      sample = file_list[idx:min(idx + batch_size, l)]
      
      for imgs,voxel,t_id in sample:
          rgba_in = Image.open(imgs)
          background = Image.new("RGB", rgba_in.size, (255, 255, 255))
          background.paste(rgba_in, mask=rgba_in.split()[3]) # 3 is the alpha channel
          rendering_image = cv2.resize(np.array(background).astype(np.float32), (224,224)) / 255.

          with open(voxel, 'rb') as f:
              volume = binvox_rw.read_as_3d_array(f)
              volume = volume.data.astype(np.float32)

          img.append(rendering_image)
          vox.append(volume)
          tax_id.append(t_id)

      img = np.asarray(img).reshape(-1,224,224,3)
      vox = np.asarray(vox).reshape(-1,32,32,32)

      yield img, vox, tax_id