import tensorflow as tf

def lr_scheduler(epoch, lr):
  decay_rate = 0.5
  decay_step = 150
  if epoch % decay_step == 0 and epoch:
    return lr * decay_rate
  return lr