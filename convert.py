from sklearn_porter import Porter
from sklearn.tree import DecisionTreeClassifier

import tensorflow as tf
import h5py
from sklearn.ensemble import RandomForestClassifier

model_filename = 'random_forest_model.h5'

with h5py.File(model_filename, 'r') as file:
    model_params = {key: value for key, value in file.attrs.items()}

loaded_rf_classifier = RandomForestClassifier(**model_params)

# Convert scikit-learn model to TensorFlow
input_dim = len(model_params['n_features'])
input_placeholder = tf.keras.layers.Input(shape=(input_dim,))
output_tensor = loaded_rf_classifier(input_placeholder)
model = tf.keras.models.Model(inputs=input_placeholder, outputs=output_tensor)

# Convert the TensorFlow model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
tflite_filename = 'random_forest_model.tflite'
with open(tflite_filename, 'wb') as tflite_file:
    tflite_file.write(tflite_model)

