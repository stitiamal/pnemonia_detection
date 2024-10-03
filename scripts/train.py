#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

data_location = 'C:/Users/ASUS/Desktop/M2_SID/MFD/chest_xray'
train_gauss = os.path.join(data_location, 'train_gauss')
test_gauss = os.path.join(data_location, 'test_gauss')
val_gauss = os.path.join(data_location, 'val_gauss')

batch_size = 32
image_size = (128, 128)

# Load and preprocess the data without data augmentation
train_generator = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
    train_gauss, target_size=image_size, batch_size=batch_size, class_mode='binary')

test_generator = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
    test_gauss, target_size=image_size, batch_size=batch_size, class_mode='binary')

val_generator = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
    val_gauss, target_size=image_size, batch_size=batch_size, class_mode='binary')

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=5, validation_data=val_generator, verbose=1)

# Make predictions on test data
test_loss, test_accuracy = model.evaluate(test_generator, verbose=2)
print(f'Test accuracy: {test_accuracy}')

