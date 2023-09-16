import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pandas as pd
import shutil
import os


# Define the path to the dataset
garbage_dataset = r'C:\Users\nicol\Documents\Python Scripts\hack the north\Garbage classification\Garbage classification'   #change path to match the one in your computer :)


datagenerator = ImageDataGenerator(
    rescale=1.0/255.0,        #normalizes the data 
    validation_split=0.2      #splits data, 20% for testing
)

batch= 32                
image_size = (150, 150)   

#generator 
train_generator = datagenerator.flow_from_directory(
    garbage_dataset,
    target_size=image_size,
    batch_size=batch,
    class_mode='categorical',     # Categorical labels (one-hot encoding)
    subset='training'             # Use the training subset
)

valid_generator = datagenerator.flow_from_directory(
    garbage_dataset,
    target_size=image_size,
    batch_size=batch,
    class_mode='categorical',
    subset='validation'
)


#defining the model 
model = Sequential([  #for sequential layers 
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),  #
    MaxPooling2D(2, 2),    #downsizes spatial dimentions
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),   #reshapes into a 1D array
    Dense(128, activation='relu'),
    Dropout(0.2),  #prevents overtraining the model, drops some of the units to avoid overfitting
    Dense(6, activation='softmax')  # Output layer with 6 classes, one for each garbage type
])


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 30  # goes through the data set 20 times
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // batch,
    epochs=epochs
)

# Save the trained model
model.save('garbage_classification_model.h5')

