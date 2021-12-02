# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd

import tensorflow as tf
from data import CustomDataloader
from model import get_model
import sklearn.metrics



EPOCHES = 100
NUM_CLASSES = 8
BATCH_SIZE = 32
IMG_HEIGTH = 448
IMG_WIDTH = 448

print("### EPOCHES: " + str(EPOCHES))
print("### BATCH SIZE: " + str(BATCH_SIZE))

### load dataset
train_loader = CustomDataloader(True, BATCH_SIZE, IMG_HEIGTH, IMG_WIDTH)
test_loader = CustomDataloader(False, BATCH_SIZE, IMG_HEIGTH, IMG_WIDTH)
NUM_BATCHES = len(train_loader)
print("### LOAD DATA")

### load model
#model = get_model()

###
#print_weights = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: print(model.layers[3].get_weights()))
#early_stopping = tf.keras.callbacks.EarlyStopping(patience=15, mode='auto', monitor='val_loss')

'''
inputs = layers.Input(shape=(IMG_HEIGTH, IMG_WIDTH, 3))
x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(rate=0.5)(x)
cls_outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=inputs, outputs=cls_outputs)
print(model.summary())
'''

#print(model.summary())



#adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
'''model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('### START TRAIN')
history = model.fit(
  x=train_loader,
  validation_data=test_loader,
  epochs=EPOCHES,
  verbose='auto',
)

model.save('/content/drive/Shareddrives/캡스톤디자인/model/resnet_based')'''

model = get_model()

loss_train = np.zeros(shape=(EPOCHES,), dtype=np.float32)
acc_train = np.zeros(shape=(EPOCHES,), dtype=np.float32)
loss_val = np.zeros(shape=(EPOCHES,))
acc_val = np.zeros(shape=(EPOCHES,))

train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.CategoricalCrossentropy()



for epoch in range(EPOCHES):
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
        train_acc_metric.update_state(y_batch_train, logits)

        # Log every 200 batches.
        if step % 10 == 0 or step == 51:    # step == 51이 최종 맞나?
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %d samples" % ((step + 1) * BATCH_SIZE))

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("===")
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()



    # Run a validation loop at the end of each epoch.
    for x_batch_test, y_batch_test in test_loader:
        val_logits = model(x_batch_test, training=False)
        # Update val metrics
        val_acc_metric.update_state(y_batch_test, val_logits)

    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()

    print("Validation acc: %.4f" % (float(val_acc),))
    print("===")

    print("Time taken: %.2fs" % (time.time() - start_time))

