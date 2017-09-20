import tensorflow as tf


def _variable_with_weight_decay(name, shape, stddev):
    var = tf.get_variable(name, shape,
                          tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    weight_decay = tf.multiply(tf.nn.l2_loss(var), name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    return var
