import os

import PIL 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pathlib
import matplotlib.pyplot as plt

from tensorflow.keras.metrics import Precision, Recall, Accuracy

data_dir = "/home/cezary/Documents/GoOnline/data/"
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print("img count",image_count)