import numpy as np
import tensorflow as tf


def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float64)
        return x

    return tf.numpy_function(f, [y_true, y_pred], tf.float64)


smooth = 1e-15


def dice_coef(y_true, y_pred, smooth=1e-5):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def dice_loss(y_true, y_pred):
    loss = 1.0 - dice_coef(y_true, y_pred)
    return loss
