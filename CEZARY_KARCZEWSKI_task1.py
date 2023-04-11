import os

import PIL 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pathlib
import matplotlib.pyplot as plt

from tensorflow.keras.metrics import Precision, Recall, Accuracy

# Load data
data_dir = "/home/cezary/Documents/GoOnline/data/"
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print("img count",image_count)

batch_size = 32
img_height = 300
img_width = 300

# Adding labels
dataset = tf.keras.utils.image_dataset_from_directory(
    directory=data_dir,
    labels='inferred', 
    label_mode='int',
    image_size=(img_height, img_width),
    batch_size=batch_size,
)
class_names = dataset.class_names
print(class_names)

# Dataset normalization 
dataset = dataset.map(lambda x, y: (x/255, y))

# Augmentation layer 
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal", input_shape=(300,300,3))
])
