import os
import random

import cv2
from tqdm import tqdm

'''
Tensorflow log levels:
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import numpy as np

TRAINING = "training"
VALIDATION = "validation"


def set_gpu():
    """
    Checks if there is a GPU available and sets the memory growth to true for every available GPU.

    :return:
    """
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def load_dataset(
        path,
        subset="training",
        validation_split=0.2,
        batch_size=32,
        img_height=480,
        img_width=640
):
    """
    Loads the image dataset from the given path.

    :param int img_width: The width of the images.
    :param int img_height: The height of the images.
    :param int batch_size: The batch size.
    :param float validation_split: The percentage of the dataset to use for validation. (Training Subset: (1 - validation_split) * dataset, Validation Subset: validation_split * dataset)
    :param str subset: subset of the dataset to load. Either "training" or "validation".
    :param str path: The path to the dataset.
    :return: dataset
    """
    path = pathlib.Path(path)

    n_dataset = len(list(path.glob("*/*")))
    n_validation = int(n_dataset * validation_split) if subset == TRAINING else (1 - validation_split) * n_dataset

    print(
        f"\n+{'-' * 14}+{'-' * 14}+\n"
        f"| {'Dataset':^12} | {n_dataset:^12} |\n"
        f"+{'-' * 14}+{'-' * 14}+\n"
        f"| {'Validation':^12} | {int(n_validation):^12} |\n"
        f"+{'-' * 14}+{'-' * 14}+\n"
        f"| {'Batch Size':^12} | {batch_size:^12} |\n"
        f"+{'-' * 14}+{'-' * 14}+\n"
        f"| {'Image Width':^12} | {f'{img_width} px':^12} |\n"
        f"+{'-' * 14}+{'-' * 14}+\n"
        f"| {'Image Height':^12} | {f'{img_height} px':^12} |\n"
        f"+{'-' * 14}+{'-' * 14}+\n"
    )

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        validation_split=validation_split,
        subset=subset,
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    return dataset


train_dataset = load_dataset("dataset/train", subset=TRAINING)

