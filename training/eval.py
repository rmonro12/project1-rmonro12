from train import build_model
import tensorflow as tf
import numpy as np

#model = build_model()
#model.load_weights('saved_models/project1_model.h5')

model = tf.keras.models.load_model('saved_models/project1_model.h5')

data = np.load("saved_datasets/data_split.npz")

X_test = data["X_test"]
y_test = data["y_test"]

test_loss, test_acc = model.evaluate(X_test, y_test)
print('\nTest accuracy:', test_acc)
model.summary()