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

class DataLoader(object):
    """
    DataLoader class. Reads  the JSON, image and voxel files\n
    :param JSON_filepath: path to dataset JSON file\n
    :param rendering_filepath: path to rendering folder of ShapeNet dataset\n
    :param voxel_filepath: path to voxel folder of ShapeNet dataset\n
    :param mode: model mode (train, test or val)\n
    :param batch_size: model input data batch size
    """

    def __init__(self, JSON_filepath, rendering_filepath, voxel_filepath, mode="train", batch_size=8, restrict=False, restriction_size=100):
        self.JSON_filepath = JSON_filepath
        self.rendering_filepath = rendering_filepath
        self.voxel_filepath = voxel_filepath
        self.mode = mode.lower()
        self.batch_size = batch_size
        self.restrict = restrict
        self.restriction_size = restriction_size

        # either all the functions can be called here or individually. But for individual call outside, mode and batch_size should be
        # passes as arguments to get_xy_paths() and data_gen() respectively
        self.taxonomy_dict = self.read_taxonomy_JSON()
        if self.mode == "inference":
            self.path_list = [self.rendering_filepath, self.voxel_filepath]
        else:
            self.path_list = self.get_xy_paths()

        self.dataset_gen = self.data_gen(self.path_list)
        self.length = len(self.path_list)

    def read_taxonomy_JSON(self):
        '''
        Read JSON file containing dataset taxonomy.\n
        :return: Un-JSON-ified dictionary
        '''
        with open(self.JSON_filepath, encoding='utf-8') as file:
            self.taxonomy_dict = json.loads(file.read())
        return self.taxonomy_dict

    def get_xy_paths(self):
        '''
        Get list of file paths for x (images) and y (voxel models).\n
        :return: List containing file path for x and corresponding y
        '''
        if self.mode == "test":
            self.logger = logger_test
        else:
            self.logger = logger_train

        self.path_list = []
        self.logger.info("Starting to read input data files for {0} phase now".format(self.mode))
        for i in range(len(self.taxonomy_dict)):
            self.logger.info(
                "Reading files of Taxonomy ID -> {0}, Class -> {1}".format(self.taxonomy_dict[i]["taxonomy_id"],
                                                                           self.taxonomy_dict[i]["taxonomy_name"]))

            for sample in self.taxonomy_dict[i][self.mode]:
                self.render_txt = os.path.join(
                    self.rendering_filepath.format(self.taxonomy_dict[i]["taxonomy_id"], sample), "renderings.txt")

                if not os.path.exists(self.render_txt):
                    self.logger.warn("Could not find file -> {0}".format(self.render_txt))
                    continue

                with open(self.render_txt, 'r') as f:
                    while (1):
                        self.value = next(f, 'end')
                        if self.value == 'end':
                            break
                        else:
                            self.img_path = os.path.join(
                                self.rendering_filepath.format(self.taxonomy_dict[i]["taxonomy_id"], sample),
                                self.value.strip('\n'))
                            self.target_path = self.voxel_filepath.format(self.taxonomy_dict[i]["taxonomy_id"], sample)
                            self.path_list.append(
                                [self.img_path, self.target_path, self.taxonomy_dict[i]["taxonomy_id"]])

        # Shuffle path list
        random.shuffle(self.path_list)  # in-place

        self.logger.info("Finished reading all the files")

        if self.restrict:
            return self.path_list[:self.restriction_size]
        else:
            return self.path_list

    def data_gen(self, file_list):
        '''
        Create generator from file path list.\n
        :param file_list: List of file paths\n
        :return: Generator object
        '''
        # print(len(file_list))
        # if self.mode == "train" or self.mode == "val" or self.mode == "test":
        if self.mode in ("train", "val", "test"):
            # Shuffle path list
            random.shuffle(file_list)  # in-place

            self.l = len(file_list)

            for idx in range(0, self.l, self.batch_size):
                self.img, self.vox, self.tax_id = [], [], []
                self.sample = file_list[idx:min(idx + self.batch_size, self.l)]

                for imgs, voxel, t_id in self.sample:
                    self.rgba_in = Image.open(imgs)
                    self.background = Image.new("RGB", self.rgba_in.size, (255, 255, 255))
                    self.background.paste(self.rgba_in, mask=self.rgba_in.split()[3])  # 3 is the alpha channel
                    self.rendering_image = cv2.resize(np.array(self.background).astype(np.float32), (224, 224)) / 255.

                    with open(voxel, 'rb') as f:
                        self.volume = binvox_rw.read_as_3d_array(f)
                        self.volume = self.volume.data.astype(np.float32)

                    self.img.append(self.rendering_image)
                    self.vox.append(self.volume)
                    self.tax_id.append(t_id)

                self.img = np.asarray(self.img).reshape(-1, 224, 224, 3)
                self.vox = np.asarray(self.vox).reshape(-1, 32, 32, 32)

                yield self.img, self.vox, self.tax_id

        elif self.mode == "inference":
            self.rgba_in = Image.open(self.path_list[0])
            if len(self.rgba_in.split()) == 4:
                self.background = Image.new("RGB", self.rgba_in.size, (255, 255, 255))
                self.background.paste(self.rgba_in, mask=self.rgba_in.split()[3])  # 3 is the alpha channel
            elif len(self.rgba_in.split()) == 3:
                self.background = self.rgba_in
            self.rendering_image = cv2.resize(np.array(self.background).astype(np.float32), (224, 224)) / 255.

            with open(self.path_list[1], 'rb') as f:
                self.volume = binvox_rw.read_as_3d_array(f)
            self.volume = self.volume.data.astype(np.float32)

            yield self.rendering_image, self.volume