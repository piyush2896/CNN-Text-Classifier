import tensorflow as tf


def binary_entropy(labels, preds):
    labels = tf.cast(labels, tf.float32)
    pos = tf.multiply(labels, tf.log(preds))
    neg = tf.multiply(tf.subtract(1, labels), tf.log(tf.subtract(1, preds)))
    return tf.reduce_mean(tf.multiply(-1, tf.add(pos, neg)))

def softmax_entropy(labels, preds):
    #labels = tf.cast(labels, tf.float32)
    loss = tf.reduce_sum(tf.multiply(-1.0, tf.multiply(labels, tf.log(preds))), axis=0)
    return tf.reduce_mean(loss)

def mse(labels, preds):
    labels = tf.cast(labels, tf.float32)
    return tf.losses.mean_squared_error(labels, preds)