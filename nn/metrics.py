import tensorflow as tf


def binary_accuracy(labels, preds):
    correct_pred = tf.cast(tf.equal(tf.greater(labels, tf.constant(0.5)), preds[-1]), tf.float32)
    return tf.reduce_mean(correct_pred)

def multiclass_accuracy(labels, preds):
    correct_pred = tf.cast(tf.equal(tf.argmax(labels, 1), tf.argmax(preds, 1)), tf.float32)
    return tf.reduce_mean(correct_pred)

def explained_variance(labels, preds):
    _, var_num = tf.nn.moments(labels - preds)
    _, var_den = tf.nn.moments(labels)
    return tf.subtract(1.0, tf.divide(var_num, var_den))