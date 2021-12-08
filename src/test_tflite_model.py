import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras



parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--file-path', type=str, required=True)
args = parser.parse_args()
MODEL = args.model
FILE = args.file_path


interpreter = tf.lite.Interpreter(model_path="../model/" + MODEL + ".tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']
output_details = interpreter.get_output_details()

image = tf.keras.preprocessing.image.load_img(FILE)
image_arr = tf.keras.preprocessing.image.img_to_array(image)
image_arr = tf.image.resize(image_arr,[input_shape[1], input_shape[2]]).numpy()        # (H, W, 3)
image_arr = image_arr/255.
image_arr = np.expand_dims(image_arr, axis=0)

print(image_arr)
with open('image_arr', 'wb') as f:
  f.write(image_arr)

print(input_details)
print(output_details)
interpreter.set_tensor(input_details[0]['index'], image_arr)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print("===")
print(output_data)
print(tf.reduce_max(output_data))
print(tf.math.argmax(output_data[0]))
