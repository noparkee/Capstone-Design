import tensorflow as tf
from tensorflow.keras.utils import Sequence

import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



### - ###
class CustomDataloader(Sequence):
    def __init__(self, train, file_type, batch_size, num_classes, img_height, img_width, shuffle=False):    
        trd = 'train_description.pkl'
        ted = 'test_description.pkl'

        if file_type == 'coin':
            trd = 'coin_' + trd
            ted = 'coin_' + ted
        elif file_type == 'paper':
            trd = 'paper_' + trd
            ted = 'paper_' + ted
        elif file_type == 'binary':
            trd = 'binary_' + trd
            ted = 'binary_' + ted

        if train:
            self.description = pd.read_pickle('../data/' + trd)
        else:
            self.description = pd.read_pickle('../data/' + ted)

        self.file_type = file_type
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle

        self.img_height = img_height
        self.img_width = img_width
        

    def __len__(self):
        return math.ceil(len(self.description) / self.batch_size)


    def __getitem__(self, index):
        batches = self.description[index * self.batch_size:(index + 1) * self.batch_size]
        images, labels = self.__get_data(batches) 
        
        return images, labels
    

    def __get_input(self, path):
        ### 이미지 전처리할거면 여기서 더 필요할 듯
        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize(image_arr,[self.img_height, self.img_width]).numpy()        # (H, W, 3)

        return image_arr/255.


    def __get_output(self, label):
        #print(label)
        #print(tf.keras.utils.to_categorical(label, num_classes=self.num_classes))
        #input()
        return tf.keras.utils.to_categorical(label, num_classes=self.num_classes)


    def __get_data(self, batches):
        # Generates data containing batch_size samples
        path_batch = batches['path']
        if self.file_type == 'binary':
            label_batch = batches['money_num']
        else:
            label_batch = batches['label_num']

        #images = np.asarray()
        #labels = np.asarray()       
        #labels = np.asarray(label_batch)                                        
        
        images = tf.convert_to_tensor([self.__get_input(x) for x in path_batch])
        labels = tf.convert_to_tensor([self.__get_output(y) for y in label_batch])  # (BATCH_SIZE, NUM_CLASSES) --> categorical_crossentropy
        #labels = tf.convert_to_tensor(label_batch)     # (BATCH_SIZE, ) --> sparse_categorical_crossentropy
        
        return images, labels
    