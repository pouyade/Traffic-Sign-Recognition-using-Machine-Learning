from TrafficSignSystem.Utilities import *
from matplotlib import pyplot as plt
import numpy
import tensorflow as tf


model = tf.keras.models.load_model('traffic.model')
print("Loading Test Datasets...")
test_images,test_labels = TrafficTestData()
print("Test Dataset Loaded.")


test_images = numpy.array(test_images)
test_labels = numpy.array(test_labels)

res = model.predict(test_images)
print("Train Dataset Loaded.")

plt.figure(figsize=(10,10))
for i in range(50):
    plt.subplot(5,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(SIGN_LABELS[int(test_labels[i])])
plt.show()