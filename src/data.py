import tensorflow as tf
from tensorflow.keras.utils import Sequence
import math
import pandas as pd
import numpy as np



class CustomDataloader(Sequence):
    def __init__(self, train, batch_size, shuffle=False):    
        if train:
            self.description = pd.read_pickle('../data/train_description.pkl')
        else:
            self.description = pd.read_pickle('../data/test_description.pkl')

        self.batch_size = batch_size
        self.shuffle=shuffle


    def __len__(self):
        return math.ceil(len(self.description) / self.batch_size)


    def __getitem__(self, index):
        batches = self.description[index * self.batch_size:(index + 1) * self.batch_size]
        images, labels = self.__get_data(batches) 
        
        return images, labels
    

    def __get_input(self, path):
        img_height = 256
        img_width = 256

        ### 이미지 전처리할거면 여기서 더 필요할 듯
        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize(image_arr,[img_height, img_width]).numpy()

        return image_arr/255.


    def __get_output(self, label, num_classes=8):
        return tf.keras.utils.to_categorical(label, num_classes=num_classes)


    def __get_data(self, batches):
        # Generates data containing batch_size samples
        path_batch = batches['path']
        label_batch = batches['label_num']

        images = np.asarray([self.__get_input(x) for x in path_batch])
        #labels = np.asarray([self.__get_output(y) for y in label_batch])       # (BATCH_SIZE, NUM_CLASSES)
        labels = np.asarray(label_batch)                                        # (BATCH_SIZE, )
        
        return images, labels
