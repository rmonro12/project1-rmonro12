# TensorFlow and tf.keras
from xml.parsers.expat import model
import tensorflow as tf
import keras
from keras import Input, layers, Sequential
from sklearn.model_selection import train_test_split

# Helper libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import pathlib

L2_RATE = 1e-4 # L2 regularization rate
DO_RATE = 0.20 # Dropout rate

# Create model architecture
def build_model():
  model = tf.keras.Sequential ([
    Input(shape=(176, 144, 3)),
    layers.SeparableConv2D(32, kernel_size=(3, 3), activation='relu', padding='same', strides=(2,2), kernel_regularizer=tf.keras.regularizers.l2(L2_RATE)),
    layers.BatchNormalization(),
    layers.SeparableConv2D(64, kernel_size=(3, 3), activation='relu', padding='same', strides=(2,2), kernel_regularizer=tf.keras.regularizers.l2(L2_RATE)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.BatchNormalization(),
    layers.SeparableConv2D(128, kernel_size=(3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_RATE)),
    layers.BatchNormalization(),
    layers.SeparableConv2D(128, kernel_size=(3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_RATE)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.BatchNormalization(),
    layers.SeparableConv2D(256, kernel_size=(3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_RATE)),
    layers.BatchNormalization(),
    layers.SeparableConv2D(256, kernel_size=(3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_RATE)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
  ])
  return model

# Define dataset parameters
IMG_SIZE = (176, 144)
data_dir = pathlib.Path("dataset")
images = []
labels = []

for class_name in ["not_a_shoe", "shoe"]:
    class_dir = data_dir / class_name
    label = 0 if class_name == "not_a_shoe" else 1
    
    for img_path in class_dir.glob("*"):
        img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
        img = tf.keras.utils.img_to_array(img)
        images.append(img)
        labels.append(label)

X = np.array(images) / 255.0   # Normalize
y = np.array(labels)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, # 80/20 train/test
    stratify=y,
    random_state=42
)

# Save for use in eval.py
np.savez("saved_datasets/data_split.npz", X_test=X_test, y_test=y_test)

# Build and compile model
model = build_model()
model.compile(optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

# Start training
train_hist = model.fit(X_train, y_train, epochs=2) # Control # of epochs for training here

# Save model weights
model.save('saved_models/project1_model.h5')
