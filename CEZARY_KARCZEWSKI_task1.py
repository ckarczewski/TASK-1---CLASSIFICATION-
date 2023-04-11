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