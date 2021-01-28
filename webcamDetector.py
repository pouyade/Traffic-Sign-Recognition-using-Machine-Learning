from TrafficSignRecognition import TrafficSignRecogniser
import cv2
from Config import Config

vc = cv2.VideoCapture(0)
detector = TrafficSignRecogniser()

while(1):
    rval, img = vc.read()
    # img = cv2.imread(Config.DATA_SET_PATH + Config.DETECT_DATA_SET_PATH + "00157.ppm",1)
    img  = cv2.resize(img,(700,450))

    detector.RunImage(img, False)
    # newimage  = cv2.resize(detector.drawedImage.copy(),(600,450))
    cv2.imshow("img",detector.drawedImage)
    cv2.moveWindow("img",0,0)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break