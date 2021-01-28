import tensorflow as tf
from TrafficSignSystem.Utilities import *
import numpy
import cv2

from TrafficSignSystem.Config import *

model = tf.keras.models.load_model('traffic.model')

img = load_image_for_cnn("q.jpg",(110,560),(240,711))
img = [numpy.array(img)]
res = model.predict([img])
cv2.imshow("t",img[0])
print(SIGN_LABELS[biggestIndex(res[0])])
cv2.waitKey(0)