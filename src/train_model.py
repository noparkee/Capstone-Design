import sys
import random
import numpy as np
import argparse
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras import optimizers

from data import CustomDataloader

import os



ITER = 100
NUM_CLASSES = 8
BATCH_SIZE = 32
IMG_HEIGTH = 448
IMG_WIDTH = 448

print("### ITERATION: " + str(ITER))
print("### BATCH SIZE: " + str(BATCH_SIZE))

#tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

### load dataset
train = CustomDataloader(True, BATCH_SIZE, IMG_HEIGTH, IMG_WIDTH)
test = CustomDataloader(False, BATCH_SIZE, IMG_HEIGTH, IMG_WIDTH)
print("### LOAD DATA")

### model
model = Sequential([
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(NUM_CLASSES)
])

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
print('### LOAD MODEL')

print('### START TRAIN')
history = model.fit(
  train,
  validation_data=test,
  epochs=ITER,
  verbose='auto'
)