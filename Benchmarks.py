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


class Benchmark:

    def __init__(self, use_gpu=False, model=None):
        """
        This class is used to perform different benchmarks on the model using GPU or CPU

        :param bool use_gpu: Whether to use GPU or CPU
        :param model: The model to benchmark
        TODO: Change model type
        """

        self.use_gpu = use_gpu

        if self.use_gpu:
            self.set_gpu()

        self.batch_size_benchmark()

    def set_gpu(self):
        print("Setting GPU...")

    def batch_size_benchmark(self, batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256]):
        """
        Measures the time it takes to train 1 epoch of the model with different batch sizes

        :param list of int batch_sizes: Batch sizes to benchmark
        :return:
        """
        print("Starting batch size benchmark...")
        times = []
        for batch_size in batch_sizes:
            print("Benchmarking batch size: " + str(batch_size))
            start_time = time.time()
            self.train_model(batch_size=batch_size)
            end_time = time.time()
            times.append(end_time - start_time)
            print("Time elapsed: " + str(datetime.timedelta(seconds=end_time - start_time)))
            print("")

