import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        ### warning log 안 뜨게
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras



parser = argparse.ArgumentParser()
parser.add_argument('--binary-model', type=str, required=True)
parser.add_argument('--classifier-model', type=str, required=True)
parser.add_argument('--file-path', type=str, required=True)
args = parser.parse_args()

BINARY_MODEL = args.binary_model
CLASSIFIER_MODEL = args.classifier_model
FILE = args.file_path

LABEL_TO_NAME = ['10won', '100won', '1000won', '10000won', '50won', '500won', '5000won', '50000won']    


### money or not
binary = tf.lite.Interpreter(model_path=BINARY_MODEL)
binary.allocate_tensors()
binary_input_details = binary.get_input_details()
binary_input_shape = binary_input_details[0]['shape']
binary_output_details = binary.get_output_details()

### make input file
image = tf.keras.preprocessing.image.load_img(FILE)
image_arr = tf.keras.preprocessing.image.img_to_array(image)
image_arr = tf.image.resize(image_arr,[binary_input_shape[1], binary_input_shape[2]]).numpy()        # (H, W, 3)
image_arr = image_arr/255.
image_arr = np.expand_dims(image_arr, axis=0)

### forward model
binary.set_tensor(binary_input_details[0]['index'], image_arr)
binary.invoke()


binary_output = binary.get_tensor(binary_output_details[0]['index'])
print("===")
print(binary_output)
binary_prob = tf.reduce_max(binary_output)
binary_class = tf.math.argmax(binary_output[0])

if binary_class == 1:       # 화폐 일 때
    ### classifier
    classifier = tf.lite.Interpreter(model_path=CLASSIFIER_MODEL)
    classifier.allocate_tensors()
    classifier_input_details = classifier.get_input_details()
    classifier_input_shape = classifier_input_details[0]['shape']
    classifier_output_details = classifier.get_output_details()
    
    ### forward model
    classifier.set_tensor(classifier_input_details[0]['index'], image_arr)
    classifier.invoke()

    classifier_output = classifier.get_tensor(classifier_output_details[0]['index'])
    print(classifier_output)
    classifier_prob = tf.reduce_max(classifier_output)
    classifier_class = tf.math.argmax(classifier_output[0])
    print(LABEL_TO_NAME[classifier_class], classifier_prob)

else:
    print("Not money.")
