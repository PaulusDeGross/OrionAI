import time, datetime
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
import numpy as np


def benchmark(GPU):
    """
    Benchmark for the given GPU.

    :param str GPU: The GPU to benchmark.
    :return: time_delta, v_ram peak: Used to calculate the score.
    """
    GPU_name = GPU.replace("/physical_device:", "")
    print("\n[ + ] Benchmarking GPU: " + GPU_name)
    try:
        load_gpu(GPU)
    except RuntimeError as e:
        return "GPU not available.", "GPU not available."


    start_time = time.time()
    n_classes = 10
    n_samples = 3000000
    batch_size = n_samples // 20
    input_units = 100
    n_epochs = 2

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_units,)),
        tf.keras.layers.Dense(2500, activation="relu"),
        tf.keras.layers.Dense(2500, activation="relu"),
        tf.keras.layers.Dense(250, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax")
    ])

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

    input = np.random.random((n_samples, input_units))
    gtt = np.random.randint(0, n_classes, n_samples)
    dataset = tf.data.Dataset.from_tensor_slices((input, gtt)).batch(batch_size)

    model.fit(dataset, epochs=n_epochs)
    end_time = time.time()
    td = str(datetime.timedelta(seconds=end_time - start_time)).split(":")

    v_ram = tf.config.experimental.get_memory_info(GPU_name)

    return td, v_ram


def get_gpus():
    """
    Checks if there is a GPU available and sets the memory growth to true for every available GPU.

    :return:
    """
    gpus = tf.config.list_physical_devices("GPU")
    return gpus


def load_gpu(GPU):
    """
    Checks if there is a GPU available and sets the memory growth to true for every available GPU.

    :param GPU:
    :return:
    """
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        for gpu in gpus:
            if gpu.name == GPU:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(e)


if __name__ == "__main__":
    print("\n[ + ] Starting Benchmark...\n")
    GPUs = get_gpus()
    print(f"[ + ] Found {len(GPUs)} GPU(s).")
    for GPU in range(len(GPUs)):
        time_delta, v_ram = benchmark(GPUs[GPU].name)

        print(
            f"+{'-' * 14}+{'-' * 14}+\n"
            f"| {'Time Delta':^12} | {time_delta[0]}h {time_delta[1]}m {time_delta[2]}s |\n"
            f"+{'-' * 14}+{'-' * 14}+\n"
            f"| {'vRAM Peak':^12} | {v_ram / 1024 / 1024 / 1024:^12.2f} GB |\n"
            f"+{'-' * 14}+{'-' * 14}+\n"
        )
