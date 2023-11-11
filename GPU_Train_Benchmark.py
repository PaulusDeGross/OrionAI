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


def benchmark(GPU, n_epochs=5):
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

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_units,)),
        tf.keras.layers.Dense(2500, activation="relu"),
        tf.keras.layers.Dense(2500, activation="relu"),
        tf.keras.layers.Dense(250, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax")
    ])

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

    _input = np.random.random((n_samples, input_units))
    gtt = np.random.randint(0, n_classes, n_samples)
    dataset = tf.data.Dataset.from_tensor_slices((_input, gtt)).batch(batch_size)

    model.fit(dataset, epochs=n_epochs)
    end_time = time.time()
    td = str(datetime.timedelta(seconds=end_time - start_time)).split(":")

    v_ram = tf.config.experimental.get_memory_info(GPU_name)

    score = ((v_ram["peak"] / 1024) / (float(td[0]) * 3600 + float(td[1]) * 60 + float(td[2])) * n_epochs) / 10000
    print(td, v_ram["peak"], score)

    return td, v_ram, score


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

    :param str GPU: The GPU to load.
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
        time_delta, v_ram, score = ['0', '01', '02.957258'], {"peak": 7900507904}, 12.254885290588735 # benchmark(GPUs[GPU].name, 1)

        print(
            "+" + "-" * 14 + "+" + "-" * 15 + "+\n"
            f"| {'Time Delta':^12} | {time_delta[0]}h {time_delta[1]}m {round(float(time_delta[2]), 3)}s |\n"
            "+" + "-" * 14 + "+" + "-" * 15 + "+\n"
            f"| {'vRAM Peak':^12} | {round(v_ram['peak'] / 1024 / 1024 / 1024, 2):^10} GB |\n"
            "+" + "-" * 14 + "+" + "-" * 15 + "+\n"
            "+" + "-" * 14 + "+" + "-" * 15 + "+\n"
            f"| {'Score':^12} | {round(score*10, 2):^13} |\n"
            "+" + "-" * 14 + "+" + "-" * 15 + "+\n\n"
        )
