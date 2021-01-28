import cv2
from Config import Config
import Utilities
import numpy as np

from Detection.PreProccessor import PreProccessor
from Detection.SignDetector import SignDetector

FileNumber = 157
MASK_CAPTION = "red"
def nothing(x):
    pass


window_width = 800
window_height = 660


#Window Names and locations
Windowname1 = "ColorPicker-POUYA-Image-Result"
Windowname2 = "ColorPicker-POUYA2-Masks"
Windowname3 = "ColorPicker-POUYA2-Controlls"

currentfilename = Utilities.createImageName(FileNumber)
ImagePath = Config.DETECT_TRAIN_DATA_SET_PATH + currentfilename
cv2.namedWindow(Windowname1)
cv2.namedWindow(Windowname2)
cv2.namedWindow(Windowname3)
cv2.moveWindow(Windowname1,10,10)
cv2.moveWindow(Windowname2,10,300)
cv2.moveWindow(Windowname3,window_width + 20,10)

img= cv2.imread(ImagePath,1)
imgOrginal = img.copy()
imgOrginal2 = img.copy()
# imgOrginal = PreProccessor.Clahe(imgOrginal)
imgOrginal = PreProccessor.Normalize(imgOrginal)
imgOrginal2 = PreProccessor.Normalize2(imgOrginal2)
imgOrginal = PreProccessor.BlurImage(imgOrginal)
imgOrginal2 = PreProccessor.BlurImage(imgOrginal2)
img = Utilities.ResizeByWidth(img,window_width/2)
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)


#Creating Trackbars
cv2.createTrackbar('ImageFile', Windowname3,FileNumber,599,nothing)

masks = Config.MASKS[MASK_CAPTION]
COUNT_OF_MASKS = len(masks)
for i in range(COUNT_OF_MASKS):
    currentmask = masks[i]
    windowcurrent=Windowname3 + "MASK#" + str(i)
    cv2.namedWindow(windowcurrent)
    cv2.createTrackbar('lh', windowcurrent,currentmask[0][0],255,nothing)
    cv2.createTrackbar('ls', windowcurrent,currentmask[0][1],255,nothing)
    cv2.createTrackbar('lv', windowcurrent,currentmask[0][2],255,nothing)
    cv2.createTrackbar('hh', windowcurrent,currentmask[1][0],255,nothing)
    cv2.createTrackbar('hs', windowcurrent,currentmask[1][1],255,nothing)
    cv2.createTrackbar('hv', windowcurrent,currentmask[1][2],255,nothing)


while(1):
    filename = cv2.getTrackbarPos('ImageFile', Windowname3)
    if(not currentfilename == Utilities.createImageName(filename)):
        currentfilename = Utilities.createImageName(filename)
        ImagePath = Config.DETECT_TRAIN_DATA_SET_PATH + currentfilename
        img = cv2.imread(ImagePath, 1)
        imgOrginal = img.copy()
        # imgOrginal = PreProccessor.Clahe(imgOrginal)
        imgOrginal = PreProccessor.Normalize(imgOrginal)
        imgOrginal = PreProccessor.BlurImage(imgOrginal)
        img = Utilities.ResizeByWidth(img, window_width / 2)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    res = None
    masks = {'red':[]}
    for i in range(COUNT_OF_MASKS):
        windowcurrent = Windowname3 + "MASK#" + str(i)
        lh = cv2.getTrackbarPos('lh', windowcurrent)
        ls = cv2.getTrackbarPos('ls', windowcurrent)
        lv = cv2.getTrackbarPos('lv', windowcurrent)
        hh = cv2.getTrackbarPos('hh', windowcurrent)
        hs = cv2.getTrackbarPos('hs', windowcurrent)
        hv = cv2.getTrackbarPos('hv', windowcurrent)
        lower_red1 = np.array([lh,ls,lv])
        upper_red1 = np.array([hh,hs,hv])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        # res1 = cv2.bitwise_and(img, img, mask=mask1)
        if(res is None):
            res = mask1
        else:
            res = res + mask1
        masks['red'].append([lower_red1,upper_red1,True])

    imgOrginaledited = Utilities.applyMask(imgOrginal.copy(),imgOrginal.copy(),imgOrginal2.copy(),masks)
    # imgOrginaledited = Utilities.MulticvtColor(imgOrginaledited, cv2.COLOR_BGR2GRAY)
    # self.editedImage = PreProccessor.RemoveSmallObjects(self.editedImage)
    # imgOrginaledited = PreProccessor.Thresholding(imgOrginaledited)
    _, _,_, drawed = SignDetector.detect(imgOrginal,imgOrginaledited,None)
    imgOrginaledited = Utilities.ResizeByWidth(imgOrginaledited[0],window_width/2)
    mergeup = np.concatenate((np.array(res), np.array(res)), axis=1)
    mergedown = np.concatenate((np.array(mask1), np.array(imgOrginaledited)), axis=1)
    drawed = Utilities.ResizeByWidth(drawed,window_width/2)

    cv2.imshow(Windowname1,mergeup)
    cv2.imshow(Windowname2,mergedown)
    cv2.imshow(Windowname3,drawed)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()