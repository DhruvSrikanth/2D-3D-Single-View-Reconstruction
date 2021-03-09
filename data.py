# ----------------------------------------------Import required Modules----------------------------------------------- #

import os
import json
import numpy as np

import cv2
from PIL import Image
import binvox_rw

import logger as log

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
    path_list = []
    log.log_status(3, "Starting to read input data files for {} phase now".format(mode))
    for i in range(len(taxonomy_dict)):
        log.log_status(3, "Reading files of Taxonomy ID: {}, Class {}".format(taxonomy_dict[i]["taxonomy_id"],
                                                                             taxonomy_dict[i]["taxonomy_name"]))
        for sample in taxonomy_dict[i][mode]:
            render_txt = os.path.join(rendering_path.format(taxonomy_dict[i]["taxonomy_id"], sample), "renderings.txt")
            if not os.path.exists(render_txt):
                log.log_status(2, "Could not find file: {}".format(render_txt))
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

    log.log_status(3, "Finished reading all the files")
    return path_list

# TODO: look at data augmentation because there is a class imbalance of images (Ask Sir)
# ----> He said check with and without to see if it is required after a performance comparison but thinks it will not be
#       required since classification is not being done in this case

def tf_data_generator(file_list):
    '''
    Create generator from file path list.\n
    :param file_list: List of file paths\n
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