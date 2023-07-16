import numpy as np  # to work with arrays
import matplotlib.pyplot as plt  # to plot
import streamlit as st  # to build UI

from PIL import Image  # to work with images

# for the actual machine learning
import tensorflow as tf
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.utils import to_categorical

# 1) data preparation

# loading data
# This line loads the CIFAR-10 dataset, which consists of images classified into 10 different categories.
# CIFAR-10 already has training data and test/validation data.
# Therefore, the dataset is split into training data (X_train and y_train) and validation data (X_val and y_val).
(X_train, y_train), (X_val, y_val) = cifar10.load_data()

# preprocessing the data
# breaking down rgb value of 255 in a range of 0 to 1 (pre-processing)
# This line divides the training data by 255 to normalize the pixel values. By dividing by 255, the RGB values of each
# pixel are rescaled to a range between 0 and 1. This normalization step ensures that all pixel values fall within the
# same range, making it easier for the model to learn.
X_train = X_train / 255
X_val = X_val / 255

# one hot encoding
# In CIFAR-10,we usually classify the data into 10 categories and the labels might be integers ranging from 0 to 9.
# One-hot encoding converts these integer labels into binary vectors of 0s and 1s.
# One-hot encoding converts these integer labels into binary vectors of 0s and 1s.
# By applying one-hot encoding, the model is prevented from assuming any inherent numerical relationship between the
# classes. It eliminates any potential bias that could arise due to numerical representation, ensuring that the model
# treats all classes equally and independently.
y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)

# 2) model definition

# creating a sequential model (neural network model as a sequence or a linear stack of layers)
model = Sequential([
    # first layer(input) - flat layer - flattens the input data into 32x32 (1024) image with 3 colour channels (rgb)
    Flatten(input_shape=(32, 32, 3)),
    # second layer(training) - dense layer - creating a neural network this 1000 neurons
    Dense(1000, activation='relu'),
    # third layer(output) - softmax and 10 are used, these indicate categorical classification (into 10 categories)
    Dense(10, activation='softmax'),
])

# 3) model compilation

# loss
# 'categorical_crossentropy': This parameter specifies the loss function used during training.
# For multi-class classification problems with integer labels (as in CIFAR-10),
# the 'categorical_crossentropy' loss function is commonly used.

# optimizer
# This parameter determines the optimization algorithm used to update the model's weights during training.
# The algorithm used here is the 'adam'.

# metrics
# These metrics provide quantitative insights into how well the model is performing on a given task,
# such as classification, regression, or clustering.
# ['accuracy']: Accuracy measures the proportion of correctly classified samples or predictions.
# It is the ratio of correct predictions to the total number of predictions made.
# Accuracy provides an overall measure of how well the model performs.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4) model training

# batch_size=64:  This parameter determines the number of samples the model will process before updating its weights.
# In this case, the model will process 64 samples.

# epochs: This parameter specifies the number of times the model will iterate over the entire training data during the
# training process (here, 10).
model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_val, y_val))

model.save('cifar10_model.h5')
