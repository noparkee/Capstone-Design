import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Flatten, Dropout, Conv2D, MaxPooling2D, Input, Reshape, Permute, Dot, Softmax
from tensorflow_addons.layers import AdaptiveAveragePooling2D
from tensorflow.keras import optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.python.keras.backend import reshape, softmax
from tensorflow.python.keras.layers.core import Permute
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


def temp_model(img_height, img_width, num_classes):
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
    x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)        # (30, 30, 128)
    
    ### attention
    shape_x = x.shape
    query = Conv2D(filters=64, kernel_size=1)(x)        # (30, 30, 64)
    query = Reshape((-1, query.shape[-1]))(query)       # (C_o, H*W)
    query = Permute((2, 1))(query)
    
    key = Conv2D(filters=64, kernel_size=1)(x)          # (30, 30, 64)
    key = Reshape((-1, key.shape[-1]))(key)             # (H*W, C_o)

    value = Conv2D(filters=shape_x[-1], kernel_size=1)(x)
    value = Reshape((-1, value.shape[-1]))(value)       # (H*W, C_i)

    energy = Dot(axes=(1, 2))([query, key])             # (H*W, H*W)
    attention = Softmax()(energy)                       # (H*W, H*W)
    
    out = Dot(axes=(1, 2))([value, attention])          # (C_i, H*W)
    out = Permute((2, 1))(out)
    out = Reshape((shape_x[1], shape_x[2], shape_x[3]))(out)
    sum_out = 0.5*out + 0.5*x
    ###

    x = MaxPooling2D()(sum_out)                               # (15, 15, 128)
    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)        # (15, 15, 256)
    x = MaxPooling2D()(x)                               # (7, 7, 256)
    features = Flatten()(x)

    ### - classifier
    x = Dense(128, activation='relu')(features)
    x = Dropout(rate=0.3)(x)
    cls_outputs = Dense(num_classes, activation=tf.nn.softmax, name='classifier_softmax')(x)

    ### - build model
    model = keras.Model(inputs=inputs, outputs=cls_outputs)

    return model


def temp2_model(img_height, img_width, num_classes):
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
    x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)        # (30, 30, 128)
    
    ### attention
    shape_x = x.shape
    query = Conv2D(filters=64, kernel_size=1)(x)        # (30, 30, 64)
    query = Reshape((-1, query.shape[-1]))(query)       # (C_o, H*W)
    query = Permute((2, 1))(query)
    
    key = Conv2D(filters=64, kernel_size=1)(x)          # (30, 30, 64)
    key = Reshape((-1, key.shape[-1]))(key)             # (H*W, C_o)

    value = Conv2D(filters=shape_x[-1], kernel_size=1)(x)
    value = Reshape((-1, value.shape[-1]))(value)       # (H*W, C_i)

    energy = Dot(axes=(1, 2))([query, key])             # (H*W, H*W)
    attention = Softmax()(energy)                       # (H*W, H*W)
    
    out = Dot(axes=(1, 2))([value, attention])          # (C_i, H*W)
    out = Permute((2, 1))(out)
    out = Reshape((shape_x[1], shape_x[2], shape_x[3]))(out)
    sum_out = 0.3*out + 0.7*x
    
    features = AdaptiveAveragePooling2D(output_size=(1,1))(sum_out)
    features = Reshape((-1, ))(features)
    ###

    #x = MaxPooling2D()(sum_out)                               # (15, 15, 128)
    #x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)        # (15, 15, 256)
    #x = MaxPooling2D()(x)                               # (7, 7, 256)
    #features = Flatten()(x)

    ### - classifier
    x = Dense(64, activation='relu')(features)
    x = Dropout(rate=0.3)(x)
    cls_outputs = Dense(num_classes, activation=tf.nn.softmax, name='classifier_softmax')(x)

    ### - build model
    model = keras.Model(inputs=inputs, outputs=cls_outputs)

    return model
