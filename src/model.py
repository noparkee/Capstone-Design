import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Flatten, Dropout, Conv2D, MaxPooling2D, Input
from tensorflow.keras import optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers.pooling import MaxPool2D



def get_model(img_height, img_width, num_classes):
    ### - input layer
    inputs = Input(shape=(img_height, img_width, 3))

    ### - featurizer
    #resnet = ResNet50(include_top=False, weights=None, input_tensor=inputs, pooling='avg')
    #features = resnet.output

    ### - featurizer
    x = Conv2D(filters=16, kernel_size=3, activation='relu')(inputs)
    x = MaxPooling2D()(x)        # pool_size=(2, 2), strides=2, padding='same'
    x = Conv2D(filters=32, kernel_size=3, activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(filters=64, kernel_size=3, activation='relu')(x)
    #x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling2D()(x)
    features = Flatten()(x)

    ### - classifier
    x = Dense(128, activation='relu')(features)
    x = Dropout(rate=0.3)(x)
    cls_outputs = Dense(num_classes, activation=tf.nn.softmax, name='classifier_softmax')(x)

    ### - build model
    model = keras.Model(inputs=inputs, outputs=cls_outputs)

    return model
