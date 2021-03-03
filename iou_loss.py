import tensorflow as tf
import numpy as np

#Implementation 1
def calc_iou_loss1(y_true, y_pred):
  import tensorflow as tf

  y_true = tf.convert_to_tensor(y_true)
  y_pred = tf.convert_to_tensor(y_pred)

  _volume = tf.cast(tf.math.greater_equal(y_pred, 0.3), dtype = tf.float32)
  # print(_volume)

  a = tf.math.multiply(y_true, _volume)
  b = tf.math.reduce_sum(a)
  intersection = tf.cast(b, dtype = tf.float32)

  c = tf.math.add(_volume,y_true)
  d = tf.cast(tf.math.greater_equal(c, 1), dtype = tf.float32)
  # print(d)
  e = tf.math.reduce_sum(d)
  union = tf.cast(e, dtype = tf.float32)

  iou = (intersection / union)
    
  return iou

def calc_iou_loss2(y_true, y_pred):
  import tensorflow as tf

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

# #Implementation 2
# def get_iou(masks, predictions):
#   import tensorflow as tf
#   ious = []
#   for i in range(batch_size):
#       mask = masks[i]
#       pred = predictions[i]
#       masks_sum = tf.reduce_sum(mask)
#       predictions_sum = tf.reduce_mean(pred)
#       intersection = tf.reduce_sum(tf.multiply(mask, pred))
#       union = masks_sum + predictions_sum - intersection
#       iou = intersection / union
#       ious.append(iou)
#   return ious
#
# y_true = np.random.randint(0,2,size=(32, 32, 32)).astype(np.float32)
# y_pred = np.random.random(size=(32,32,32)).astype(np.float32)
#
# print("iou1 - {}".format(calc_iou_loss1(y_true, y_pred)))
# print("iou2 - {}".format(calc_iou_loss2(y_true, y_pred)))

# iou = get_iou(masks, predictions)
# mean_iou_loss = tf.Variable(initial_value=-tf.log(tf.reduce_sum(iou)), name='loss', trainable=True)
# train_op = tf.train.AdamOptimizer(0.001).minimize(mean_iou_loss)