from TrafficSignRecognition import TrafficSignRecogniser
import cv2
from Config import Config

# imagefile = Config.DETECT_TRAIN_DATA_SET_PATH + "00001.ppm"
imagefile = Config.DETECT_TRAIN_DATA_SET_PATH + "00157.ppm"
# imagefile = Config.DETECT_TRAIN_DATA_SET_PATH + "00080.ppm"
# imagefile = Config.DETECT_TRAIN_DATA_SET_PATH + "00090.ppm"
# imagefile = Config.DETECT_TRAIN_DATA_SET_PATH + "00093.ppm"
# imagefile = Config.DETECT_TRAIN_DATA_SET_PATH + "00095.ppm"
# imagefile = Config.DETECT_TRAIN_DATA_SET_PATH + "00202.ppm"
img = cv2.imread(imagefile,1)
# img *= 2
detector = TrafficSignRecogniser()
detector.RunImage(img)
# newimage  = cv2.resize(detector.drawedImage.copy(),(600,400))
cv2.imshow("img",detector.drawedImage.copy())
cv2.moveWindow("img",0,0)
cv2.imwrite("/Users/pouyadark/Desktop/test/red.jpg",detector.drawedImage.copy())

cv2.waitKey(0)
