from train import build_model
import tensorflow as tf

model = build_model()
model.load_weights('saved_models/project1_model.h5')



test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
model.summary()