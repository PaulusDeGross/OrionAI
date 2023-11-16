import os

'''
Tensorflow log levels:
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import tqdm
import tensorflow as tf
import matplotlib
import numpy as np

print("Tensorflow version: " + tf.__version__)
print("OpenCV version: " + cv2.__version__)
print("Numpy version: " + np.__version__)
print("Matplotlib version: " + matplotlib.__version__)
print("TQDM version: " + tqdm.__version__)