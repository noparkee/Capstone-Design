# -*- coding: utf-8 -*-
import os
import time
import datetime
import argparse
import numpy as np
import pandas as pd

import tensorflow as tf
from data import CustomDataloader
from model import get_model, temp_model, temp2_model



### default value
EPOCHES = 15
NUM_CLASSES = 8
BATCH_SIZE = 32
IMG_HEIGTH = 32        # 256 -> 224
IMG_WIDTH = 32


def save_model(ft, epoch, model):
    d = datetime.datetime.now()
    if ft == None:
        ft = ''
    FILE_NAME = d.strftime('%Y-%m-%d-%H:%M') + '_' + ft + '_' + str(epoch) + '_' + str(BATCH_SIZE) + '_' + str(IMG_HEIGTH) + 'x' + str(IMG_WIDTH)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    # tflite
    with open('../model/' + FILE_NAME + '.tflite', 'wb') as f:
        f.write(tflite_model)

    # savedmodel
    model.save('../model/' + FILE_NAME)



### changes values
parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str)
parser.add_argument('--epoches', type=int)
args = parser.parse_args()

FILE_TYPE = args.type
if FILE_TYPE == 'binary':
    NUM_CLASSES = 2
if args.epoches != None:
    EPOCHES = args.epoches


print("### EPOCHES: " + str(EPOCHES))
print("### BATCH SIZE: " + str(BATCH_SIZE))
print("### INPUT SIZE: " + str(IMG_HEIGTH))

### load dataset
train_loader = CustomDataloader(True, FILE_TYPE, BATCH_SIZE, NUM_CLASSES, IMG_HEIGTH, IMG_WIDTH)
test_loader = CustomDataloader(False, FILE_TYPE, BATCH_SIZE, NUM_CLASSES, IMG_HEIGTH, IMG_WIDTH)
NUM_BATCHES = len(train_loader)
print("### LOAD DATA")

### model load / get_model: 전체 / get_coin_model: 동전 분류 / get_paper_model: 지폐 분류
if FILE_TYPE == 'temp':
    model = temp_model(IMG_HEIGTH, IMG_WIDTH, NUM_CLASSES)
elif FILE_TYPE == 'temp2':
    model = temp2_model(IMG_HEIGTH, IMG_WIDTH, NUM_CLASSES)
else:
    model = get_model(IMG_HEIGTH, IMG_WIDTH, NUM_CLASSES)
print("### LOAD MODEL")



train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.CategoricalCrossentropy()

for epoch in range(EPOCHES):
    print("\n---------- ---------- ----------")
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_loader):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Update training metric.
        train_acc_metric.update_state(y_batch_train, logits)        # update_state: accumulate metric statistic

        # Log every 200 batches.
        if step % 10 == 0:    # step == 51이 최종 맞나?
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %d samples" % ((step + 1) * BATCH_SIZE))

        #print(y_batch_train)
        #print(logits)
        #input()

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("===")
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()



    # Run a validation loop at the end of each epoch.
    for x_batch_test, y_batch_test in test_loader:
        val_logits = model(x_batch_test, training=False)
        #print(val_logits)
        #print(tf.reduce_sum(val_logits, keepdims=True, axis=1))
        ###
        # val_logits가 output인데 아마 최대값이 0.2 미만이면 tts는 안 하는게 좋을 것 같음 아니면 0.3?
        # Update val metrics
        val_acc_metric.update_state(y_batch_test, val_logits)

    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()

    print("Validation acc: %.4f" % (float(val_acc),))
    print("===")

    print("Time taken: %.2fs" % (time.time() - start_time))
    print("---------- ---------- ----------")

    if epoch == 9:
        save_model(FILE_TYPE, epoch+1, model)

### 마지막 epoch
save_model(FILE_TYPE, EPOCHES, model)