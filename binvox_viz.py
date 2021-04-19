# ----------------------------------------------Import required Modules----------------------------------------------- #

import cv2
import matplotlib.pyplot as plt
import os

from mpl_toolkits.mplot3d import Axes3D

# ----------------------------------------------Define Binvox Visualization------------------------------------------- #

def get_volume_views(volume, save_dir):
    '''
    Get single view of volume
    :param volume: Volume to be viewed and saved
    :param save_dir: Save directory of volume view
    :return: Volume view saved
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    volume = volume.squeeze().__ge__(0.5)
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.set_aspect('auto')
    ax.voxels(volume, edgecolor="k")

    save_path = os.path.join(save_dir + '\\voxel_snapshot.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return cv2.imread(save_path)