import tensorflow as tf
from TrafficSignSystem.Utilities import *
import numpy

model = tf.keras.models.load_model(Config.RECOGNITION_MODEL_PATH)

print("Loading Test Datasets...")
test_images,test_labels = TrafficTestData()
train_images,train_labels = TrafficTrainData()
print("Test Dataset Loaded.")

train_images = numpy.array(train_images)
train_labels = numpy.array(train_labels)


test_images = numpy.array(test_images)
test_labels = numpy.array(test_labels)



test_loss, test_acc = model.evaluate(train_images, train_labels)

print('Train Data accuracy:', test_acc)
print('Train Data Lost:', test_loss)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test Data accuracy:', test_acc)
print('Test Data Lost:', test_loss)

