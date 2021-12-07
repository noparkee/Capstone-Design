import tensorflow as tf
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str)
args = parser.parse_args()
FILE = args.file

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('../model/'+FILE) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('../model/' + FILE + '.tflite', 'wb') as f:
  f.write(tflite_model)