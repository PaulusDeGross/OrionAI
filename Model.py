# TODO: Rewrite this file to use PyTorch ROCm instead of Tensorflow

import os
import pathlib
import random

import cv2
import numpy as np
from PIL import Image

'''
Tensorflow log levels:
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import matplotlib.pyplot as plt


def set_gpu(self):
    print("Setting GPU...")


def model():
    """
    Creates the model.

    :return: model
    """

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    return model


def load_model(path):
    """
    Loads the model from the given path.

    :param str path: The path to the model.
    :return: model
    """

    return tf.keras.models.load_model(path)


def get_random_image(path):
    """
    Returns a random image from the given path.

    :param str path: The Path to the root directory of the dataset.
    :return: img
    """
    img_path = random.choice(list(pathlib.Path(path).glob("*/*.png")))
    print()
    print(img_path.as_posix().split("/")[-2])
    img = np.array(Image.open(img_path.as_posix()))
    img_resized = cv2.resize(img, (640, 480))
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_resized, axis=0)
    img_batch_with_channel = np.expand_dims(img_batch, axis=-1)
    img_batch_with_channels = np.repeat(img_batch_with_channel, 3, axis=-1)
    return (img, img_batch_with_channels)


for i in range(10):
    _img, img = get_random_image("dataset/test")

    _model = load_model("benchmark_model.h5")

    index = np.argmax(_model.predict(img)[0])
    print(index)


