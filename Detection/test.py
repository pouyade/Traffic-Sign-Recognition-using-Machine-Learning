import time
from skimage import color
import cv2
from Config import Config
import numpy as np
from imutils.object_detection import non_max_suppression
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from sklearn.externals import joblib

# define the sliding window:
def sliding_window(image, stepSize,windowSize):  # image is the input, step size is the no.of pixels needed to skip and windowSize is the size of the actual window
    # slide a window across the image
    for y in range(0, image.shape[0],
                   stepSize):  # this line and the line below actually defines the sliding part and loops over the x and y coordinates
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y: y + windowSize[1], x:x + windowSize[0]])


# %%
# Upload the saved svm model:
model = joblib.load(Config.DETECTION_MODEL_PATH)

# Test the trained classifier on an image below!
start_time = time.time()

scale = 0
detections = []
filename = Config.DETECT_TRAIN_DATA_SET_PATH + "00088.ppm"
print(filename)
# read the image you want to detect the object in:
img = cv2.imread(filename)
# img = np.float32(img) / 255.0
# gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
# gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
# mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
# img = mag

# Try it with image resized if the image is too big
img = cv2.resize(img,(600, 400))  # can change the size to default by commenting this code out our put in a random number

# defining the size of the sliding window (has to be, same as the size of the image in the training data)
(winW, winH) = (32,32)
# (winW, winH) = Config.WINDOW_SIZE
windowSize = (winW, winH)
downscale = 1.5
# Apply sliding window:
for resized in pyramid_gaussian(img, downscale=downscale):  # loop over each layer of the image that you take!
    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(resized, stepSize=10, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it!
        if window.shape[0] != winH or window.shape[
            1] != winW:  # ensure the sliding window has met the minimum size requirement
            continue
        window = color.rgb2gray(window)
        window = cv2.resize(window,(64,64))
        fds = hog(window, Config.orientations, Config.pixels_per_cell, Config.cells_per_block,block_norm='L2')  # extract HOG features from the window captured
        fds = fds.reshape(1, -1)  # re shape the image to make a silouhette of hog
        pred = model.predict(fds)  # use the SVM model to make a prediction on the HOG features extracted from the window

        if pred == 1:
            if model.decision_function(
                    fds) > Config.threshold:  # set a threshold value for the SVM prediction i.e. only firm the predictions above probability of 0.6
                print("Detection:: Location -> ({}, {})".format(x, y))
                print("Scale ->  {} | Confidence Score {} \n".format(scale, model.decision_function(fds)))
                detections.append(
                    (int(x * (downscale ** scale)), int(y * (downscale ** scale)), model.decision_function(fds),
                     int(windowSize[0] * (downscale ** scale)),  # create a list of all the predictions found
                     int(windowSize[1] * (downscale ** scale))))
    scale += 1

clone = resized.copy()
for (x_tl, y_tl, cs, w, h) in detections:
    cv2.rectangle(img, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 255), thickness=2)
    cv2.putText(img,"CS:" + str(cs),(x_tl,y_tl),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),thickness=2)
rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])  # do nms on the detected bounding boxes
sc = [score[0] for (x, y, score, w, h) in detections]
print("detection confidence score: ", sc)
sc = np.array(sc)
pick = non_max_suppression(rects, probs=sc, overlapThresh=0.3)

# the peice of code above creates a raw bounding box prior to using NMS
# the code below creates a bounding box after using nms on the detections
# you can choose which one you want to visualise, as you deem fit... simply use the following function:
# cv2.imshow in this right place (since python is procedural it will go through the code line by line).
print("--- %s seconds ---" % (time.time() - start_time))

for (xA, yA, xB, yB) in pick:
    cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)
cv2.imshow("Raw Detections after NMS", img)
#### Save the images below
k = cv2.waitKey(0) & 0xFF
if k == 27:  # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('saved_image.png', img)
    cv2.destroyAllWindows()


