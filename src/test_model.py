import numpy as np
import argparse
import tensorflow as tf
from tensorflow import keras



parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--file-path', type=str, required=True)
args = parser.parse_args()
MODEL = args.model
FILE = args.file_path

loaded_model = keras.models.load_model("../model/" + MODEL)

img_height = loaded_model.input.shape[1]
img_width = loaded_model.input.shape[2]

image = tf.keras.preprocessing.image.load_img(FILE)
image_arr = tf.keras.preprocessing.image.img_to_array(image)
image_arr = tf.image.resize(image_arr,[img_height, img_width]).numpy()        # (H, W, 3)
image_arr = image_arr/255.
image_arr = np.expand_dims(image_arr, axis=0)


output = loaded_model(image_arr)
print("===")
print(output)
print(tf.reduce_max(output))
print(tf.math.argmax(output[0]))
