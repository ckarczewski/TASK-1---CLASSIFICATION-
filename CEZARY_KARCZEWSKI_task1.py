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

# Data splitting
def data_split(dataset, 
               train_split=0.8, 
               val_split=0.1, 
               test_split=0.1, 
               shuffle=True, 
               shuffle_size=10000):
    assert (train_split + val_split + test_split) == 1
    
    if shuffle:
        dataset = dataset.shuffle(shuffle_size, seed=123)
    
    train_size = int(len(dataset)*.8)
    val_size = int(len(dataset)*.1)

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size).skip(val_size)

    return train_dataset, val_dataset, test_dataset

train_dataset, val_dataset, test_dataset = data_split(dataset=dataset)

# Build model
num_classes = 4
model = keras.Sequential([
    data_augmentation,
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes),
    keras.layers.Dropout(rate=0.2)
])

# Compile model
model.compile(optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# Train
epochs=10
history = model.fit(
  train_dataset,
  validation_data=val_dataset,
  epochs=epochs
)