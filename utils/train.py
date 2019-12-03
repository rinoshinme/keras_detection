import tensorflow as tf


def get_optimizer(optim_name, lr):
    if optim_name == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    else:
        raise ValueError('optimizer not supported')

    return optimizer
