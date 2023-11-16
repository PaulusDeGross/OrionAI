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


class BatchSizeBenchmark:

    def __init__(self, model, batch_sizes, dataset):
        """
        Finds the most optimal batch size for the given model.

        :param model: Model to benchmark.
        :param batch_sizes: List of batch sizes to benchmark.
        """

        self.model = model
        self.batch_sizes = batch_sizes
