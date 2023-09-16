import tensorflow as tf

# Load the trained Keras model
model = tf.keras.models.load_model('garbage_classification_model.h5')

# Convert the Keras model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('garbage_classification_model.tflite', 'wb') as f:
    f.write(tflite_model)