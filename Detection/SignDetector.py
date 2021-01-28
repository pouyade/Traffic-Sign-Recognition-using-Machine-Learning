import Utilities
import cv2
import numpy as np
from Config import Config
from PreProccessor import PreProccessor
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
from skimage.feature import hog
from Drawer import Drawer


class SignDetector:

    # Return array of images
    @staticmethod
    # def detect(orginalimage,imgarray):
    def detect(orginalimage, imgarray, detectormodel):
        # Utilities.ShowImgArray(imgarray)
        alloutputs = []
        alllocations = []
        allnegetiveoutputs = []
        allScores = []
        drawedimg = orginalimage.copy()

        for img in imgarray:
            # output, locations, drawedimg = SignDetector.DetectRois(img.copy(), orginalimage.copy())
            output, locations, drawedimg = SignDetector.DetectRois(img.copy(), orginalimage.copy())
            for ii in range(len(output)):
                cuimg = output[ii]
                if (not detectormodel is None) & Config.CHECK_HOG :
                    sc = SignDetector.CheckHogForSign(cuimg, detectormodel)
                    if (sc > 0):
                        alloutputs.append(cuimg)
                        allScores.append(sc)
                        alllocations.append(locations[ii])
                    else:
                        # imgs,locs = SignDetector.DetectCircles(cuimg)
                        # if(len(imgs)>0):
                        allnegetiveoutputs.append(cuimg)



                else:
                    alloutputs.append(cuimg)
                    allScores.append([0])
                    alllocations.append(locations[ii])
        # drawedimg = Drawer.DrawSigns(orginalimage,loc,caption,classid,confedece,detectscore):
        if(Config.DEBUG_SAVE_SIGNS):
            Utilities.saveSigns(alloutputs,Config.SIGN_POSTIVE_RESTUL_PATH)
            Utilities.saveSigns(allnegetiveoutputs,Config.SIGN_NEGETIVE_RESTUL_PATH)
        alloutputs, alllocations, allScores,removedLocations = SignDetector.eliminateMultiDetect(alloutputs, alllocations, allScores)
        Drawer.DrawRemoveds(drawedimg,removedLocations)
        return alloutputs, alllocations, allScores, drawedimg

    @staticmethod
    def DetectHog(img, orginalimage, detectionmodel):
        drawed = orginalimage.copy()
        edited = img.copy()
        output = []
        locations = []

        edited = PreProccessor.Sobel(edited)
        scale = 0
        detections = []
        for resized in pyramid_gaussian(edited,
                                        downscale=Config.downscale):  # loop over each layer of the image that you take!
            # loop over the sliding window for each layer of the pyramid
            for (x, y, window) in Utilities.sliding_window(resized, stepSize=10, windowSize=Config.WINDOW_SIZE):
                # if the window does not meet our desired window size, ignore it!
                if window.shape[0] != Config.WINDOW_SIZE_HEIGHT or window.shape[
                    1] != Config.WINDOW_SIZE_WIDTH:  # ensure the sliding window has met the minimum size requirement
                    continue
                # window = color.rgb2gray(window)
                window = cv2.resize(window, (64, 64))
                window = PreProccessor.Sobel(window)
                fds = hog(window, Config.orientations, Config.pixels_per_cell, Config.cells_per_block,
                          block_norm='L2')  # extract HOG features from the window captured
                fds = fds.reshape(1, -1)  # re shape the image to make a silouhette of hog
                pred = detectionmodel.predict(
                    fds)  # use the SVM model to make a prediction on the HOG features extracted from the window

                if (pred == 1):
                    detections.append((int(x * (Config.downscale ** scale)), int(y * (Config.downscale ** scale)),
                                       detectionmodel.decision_function(fds),
                                       int(Config.WINDOW_SIZE[0] * (Config.downscale ** scale)),
                                       # create a list of all the predictions found
                                       int(Config.WINDOW_SIZE[1] * (Config.downscale ** scale))))
            scale += 1

        for (x_tl, y_tl, cs, w, h) in detections:
            cv2.rectangle(drawed, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 255), thickness=2)
            cv2.putText(drawed, "CS:" + str(cs), (x_tl, y_tl), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
            locations.append([x_tl, y_tl, w, h])
        rects = np.array(
            [[x, y, x + w, y + h] for (x, y, _, w, h) in detections])  # do nms on the detected bounding boxes
        sc = [score[0] for (x, y, score, w, h) in detections]
        sc = np.array(sc)
        pick = non_max_suppression(rects, probs=sc, overlapThresh=0.3)
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(drawed, (xA, yA), (xB, yB), (0, 255, 0), 2)
            cim = orginalimage[yA:yB, xA:xB]
            output.append(cim)

        return output, locations, drawed

    @staticmethod
    def DetectRois(img, orginalimage):
        orimg = orginalimage.copy()
        drawed = orimg.copy()
        output = []
        locations = []
        removedlocations = []
        removedoutputs = []
        # cv2MajorVersion = cv2.__version__.split(".")[0]
        # if int(cv2MajorVersion) >= 4:
        ctrs, hier = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # else:
        #     im2, ctrs, hier = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

        for i, ctr in enumerate(sorted_ctrs):
            x, y, w, h = cv2.boundingRect(ctr)
            area = w * h
            ratio = float(w) / float(h)

            if (area > Config.MIN_OBJECT_SIZE and (
                    Utilities.inRange(ratio, Config.SINGLE_ROI_RATIO)
                    or Utilities.inRange(ratio, Config.Duble_RIO_RATIO)
                    or Utilities.inRange(ratio, Config.Trible_RIO_RATIO)
            )):

                if (Utilities.inRange(ratio, Config.Duble_RIO_RATIO)):
                    y1 = y
                    h = h / 2
                    y2 = y1 + h
                    roi1 = orimg[y1:y1 + h, x:x + w]
                    roi2 = orimg[y2:y2 + h, x:x + w]
                    output.append(roi1)
                    locations.append([x, y1, w, h])
                    output.append(roi2)
                    locations.append([x, y2, w, h])
                elif (Utilities.inRange(ratio, Config.Trible_RIO_RATIO)):
                    h = h / 3
                    y1 = y
                    y2 = y1 + h
                    y3 = y1 + h + h
                    roi1 = orimg[y1:y1 + h, x:x + w]
                    roi2 = orimg[y2:y2 + h, x:x + w]
                    roi3 = orimg[y3:y3 + h, x:x + w]
                    output.append(roi1)
                    locations.append([x, y1, w, h])
                    output.append(roi2)
                    locations.append([x, y2, w, h])
                    output.append(roi3)
                    locations.append([x, y3, w, h])
                else:
                    roi = orimg[y:y + h, x:x + w]
                    locations.append([x, y, w, h])
                    output.append(roi)
            else:
                removedlocations.append([x, y, w, h])

        drawed = Drawer.DrawROIs(orginalimage.copy(),locations)
        if(Config.DEBUG_DRAW_DETECT_REMOVED_ROIS):
            drawed = Drawer.DrawRemoveds(drawed,removedlocations)

        return output, locations, drawed

    @staticmethod
    def DetectCircles(img):
        output = []
        locations = []
        imggrey = img.copy()
        imggrey = cv2.cvtColor(imggrey,cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(imggrey, cv2.HOUGH_GRADIENT, 1, 10,
                                   param1=30,
                                   param2=15,
                                   minRadius=50,
                                   maxRadius=255)
        if not circles is None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                x1 = i[0] - i[2]
                y1 = i[1] - i[2]
                x2 = i[0] + i[2]
                y2 = i[1] + i[2]
                imgcroped = img[y1:y2, x1:x2]
                output.append(imgcroped)
                locations.append([x1, y1, x2 - x1, y2 - y1])
        # cv2.imshow("t", img)
        # cv2.waitKey(0)
        return output, locations

    # return orimg
    @staticmethod
    def CheckHogForSign(img, detectionmodel):
        window = img.copy()
        fd = Utilities.FixImageForSVM(window)
        fds = fd.reshape(1, -1)  # re shape the image to make a silouhette of hog
        pred = detectionmodel.predict(fds)  # use the SVM model to make a prediction on the HOG features extracted from the window
        sc = detectionmodel.decision_function(fds)
        if (pred == 1 and sc > Config.threshold):  # set a threshold value for the SVM prediction i.e. only firm the predictions above probability of 0.6
            return sc
        else:
            return 0

    @staticmethod
    def eliminateMultiDetect(output, locations, scores):
        filteredoutput = output[:]
        filteredlocations = locations[:]
        removedLocations = []
        filteredscores = scores[:]
        for i in xrange(len(filteredlocations) - 1, -1, -1):
            loc1x1 = filteredlocations[i][0]
            loc1y1 = filteredlocations[i][1]
            loc1x2 = filteredlocations[i][2]
            loc1y2 = filteredlocations[i][3]
            for j in range(len(filteredlocations) - 1, -1, -1):
                if (i == j):
                    continue
                loc2x1 = filteredlocations[j][0]
                loc2y1 = filteredlocations[j][1]
                loc2x2 = filteredlocations[j][2]
                loc2y2 = filteredlocations[j][3]
                if (Utilities.inRange(loc1x1 - loc2x1, Config.MULTI_DETECT_THRESHOLD_RANGE)):
                    if (Utilities.inRange(loc1x2 - loc2x2, Config.MULTI_DETECT_THRESHOLD_RANGE)):
                        if (Utilities.inRange(loc1y1 - loc2y1, Config.MULTI_DETECT_THRESHOLD_RANGE)):
                            if (Utilities.inRange(loc1y2 - loc2y2, Config.MULTI_DETECT_THRESHOLD_RANGE)):
                                outpux = filteredoutput[i]
                                scorex = filteredscores[i]
                                try:
                                    removedLocations.append(filteredlocations[i])
                                    removedLocations.append(filteredlocations[j])
                                    del filteredlocations[i]
                                    del filteredoutput[i]
                                    del filteredscores[i]
                                    del filteredlocations[j]
                                    del filteredoutput[j]
                                    del filteredscores[j]
                                except:
                                    if(Config.DEBUG_PRINT_ERRORS):
                                        print "Exeption > eliminateMultiDetect"
                                filteredoutput.append(outpux)
                                filteredscores.append(scorex)
                                maxx1 = min(loc1x1, loc2x1)
                                maxy1 = min(loc1y1, loc2y1)
                                maxx2 = max(loc1x2, loc2x2)
                                maxy2 = max(loc1y2, loc2y2)
                                filteredlocations.append([maxx1, maxy1, maxx2, maxy2])
                                break

        return filteredoutput, filteredlocations, filteredscores , removedLocations
