import tensorflow as tf
from TrafficSignSystem import Utilities
import numpy
import cv2

from TrafficSignSystem.Config import Config

model = tf.keras.models.load_model(Config.RECOGNITION_MODEL_PATH)
print("Loading Test Datasets...")
test_images,test_labels = Utilities.TrafficTrainData()
print("Test Dataset Loaded.")


test_images = numpy.array(test_images)
test_labels = numpy.array(test_labels)

res = model.predict(test_images)
trueanswers = 0
falseanswers = 0
for i in range(len(test_images)):
    gused =Utilities.biggestIndex(res[i])
    truea = test_labels[i]
    if str(gused) == truea:
        trueanswers += 1
        continue
    falseanswers += 1
    # print("Gauesed:>"+ SIGN_LABELS[gused] +"/really:>" + SIGN_LABELS[int(truea)])
    # cv2.imshow("test",test_images[i])
    # cv2.waitKey(0)

testindex = 3000
cv2.imshow("t",test_images[testindex])
print("result("+str(len(res))+") > True:>" + str(trueanswers) + " / False:>" + str(falseanswers))
print("Result:"+str( float(trueanswers)/float(len(res)) ) + " Success/"+str( float(falseanswers)/float(len(res)) ) + " Error")
print(Config.SIGN_LABELS[Utilities.biggestIndex(res[testindex])])
cv2.waitKey(0)