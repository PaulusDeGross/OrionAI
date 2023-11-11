import os

'''
Tensorflow log levels:
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf


def model():
    """
    Creates the model.

    :return: model
    """

    model = tf.keras.models.Sequential([
        tf.keras.layers.Normalization(),
    ])