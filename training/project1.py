# TensorFlow and tf.keras
from xml.parsers.expat import model
import tensorflow as tf
import keras
from keras import Input, layers, Sequential

# Helper libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image

L2_RATE = 0
DO_RATE = 0.50

def build_model():
  model = tf.keras.Sequential ([
    Input(shape=(32, 32, 3)),
    layers.SeparableConv2D(32, kernel_size=(3, 3), activation='relu', padding='same', strides=(2,2), kernel_regularizer=tf.keras.regularizers.l2(L2_RATE)),
    layers.Dropout(DO_RATE),
    layers.SeparableConv2D(64, kernel_size=(3, 3), activation='relu', padding='same', strides=(2,2), kernel_regularizer=tf.keras.regularizers.l2(L2_RATE)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(DO_RATE),
    layers.SeparableConv2D(128, kernel_size=(3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_RATE)),
    layers.Dropout(DO_RATE),
    layers.SeparableConv2D(128, kernel_size=(3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_RATE)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(DO_RATE),
    layers.SeparableConv2D(256, kernel_size=(3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_RATE)),
    layers.Dropout(DO_RATE),
    layers.SeparableConv2D(256, kernel_size=(3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_RATE)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(DO_RATE),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
  ])
  return model

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
val_frac = 0.1
num_val_samples = int(len(train_images)*val_frac)
val_idxs = np.random.choice(np.arange(len(train_images)), size=num_val_samples, replace=False)
trn_idxs = np.setdiff1d(np.arange(len(train_images)), val_idxs)
val_images = train_images[val_idxs, :,:,:]
train_images = train_images[trn_idxs, :,:,:]
val_labels = train_labels[val_idxs]
train_labels = train_labels[trn_idxs]
train_labels = train_labels.squeeze()
test_labels = test_labels.squeeze()
val_labels = val_labels.squeeze()
input_shape  = train_images.shape[1:]
train_images = train_images / 255.0
test_images  = test_images  / 255.0
val_images   = val_images   / 255.0

model = build_model()
model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])

train_hist = model.fit(train_images, train_labels,
    validation_data=(val_images, val_labels),
    epochs=30)
model.save('saved_models/model_conv')
#test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
#print('\nTest accuracy:', test_acc)
#model.summary()