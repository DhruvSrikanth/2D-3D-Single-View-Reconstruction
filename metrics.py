# ----------------------------------------------Import required Modules----------------------------------------------- #

import numpy as np
import tensorflow as tf

# ----------------------------------------------Define Optimizer------------------------------------------------------ #

# Calculate IOU loss
def calc_iou_loss(y_true, y_pred):
    '''
    Calculate Intersection Over Union for the given batch\n
    :param y_true: Target Voxel Output\n
    :param y_pred: Predicted Voxel Output\n
    :return: IoU for batch (list)
    '''
    # y_true = tf.convert_to_tensor(y_true)
    # y_pred = tf.convert_to_tensor(y_pred)
    # print(y_true.shape)
    # print(y_pred.shape)
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
# y_true = []
# y_pred = []
# for i in range(batch_size):
#   y_true_temp = np.random.randint(0,2,size=(32, 32, 32)).astype(np.float32)
#   # y_true = np.array([y_true_temp, y_true])
#   y_true.append(y_true_temp)
#   # y_true = np.concatenate((y_true_temp, ))
#   # print(y_true.shape)
#   y_pred_temp = np.random.random(size=(32,32,32)).astype(np.float32)
#   y_pred.append(y_pred_temp)

# y_true = np.array(y_true)
# y_pred = np.array(y_pred)

# ans = calc_iou_loss(y_true, y_pred)
# # print(ans)
# print("iou - {}".format(ans))

def iou_dict_update(tax_id, iou_dict, iou):
    '''
    Update IOU dictionary for each class.\n
    :param tax_id: Class ID\n
    :param iou_dict: iou_dict Dictionary\n
    :param iou: IOU List\n
    :return: Updated IOU Dictionary
    '''
    for i, j in enumerate(tax_id):
      if j not in iou_dict:
        iou_dict[j] = {'n_samples': 0, 'iou': []}

      iou_dict[j]['n_samples'] += 1
      iou_dict[j]['iou'].append(iou[i])

    return iou_dict

def calc_mean_iou(iou_dict, mean_iou):
    '''
    Calculate mean iou for all classes based.\n
    :param iou_dict: IOU Dictionary for each class\n
    :param mean_iou: variable to append mean IOU\n
    :return: Mean IOU Dictionary
    '''
    for taxonomy_id in iou_dict:
        mean_iou[taxonomy_id] = sum(iou_dict[taxonomy_id]['iou']) / len(iou_dict[taxonomy_id]['iou'])

    return mean_iou