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

from data import CustomDataloader

import os



ITER = 100
NUM_CLASSES = 8
BATCH_SIZE = 32
print("### ITERATION: " + str(ITER))
print("### BATCH SIZE: " + str(BATCH_SIZE))

#tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

### load dataset
train = CustomDataloader(True, 32)
test = CustomDataloader(False, 32)
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

model.compile(optimizer='adam',
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