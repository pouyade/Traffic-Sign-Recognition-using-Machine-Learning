from Utilities import *
from Config import Config
import cv2

class Drawer:
    @staticmethod
    def DrawSigns(img,loc,caption,classid,confedece,detectscore):
        img = img.copy()
        width = img.shape[1]
        height = img.shape[0]
        x1 = loc[0]
        y1 = loc[1]
        x2 = loc[2]
        y2 = loc[3]

        cv2.rectangle(img, (x1, y1), (x1 + x2, y1 + y2), (0, 255, 0), 2)
        detectedsign = Config.SIGN_IMAGES[classid];
        if(x1-50 > 0 and y1 + 50 < height):
            img[y1:y1+50,x1-50:x1] = detectedsign
        elif(y1 + 50 > height and x1-50 < 0):
            img[y1-50:y1,x2:x2+50] = detectedsign
        else:
            if(y1+50 > height):
                img[y1-30:y1 + 20, x2:x2 + 50] = detectedsign
            else:
                img[y1:y1+50,x2:x2+50] = detectedsign

        # img[y1-35:y1,x1:x1+110] = [255,0,0]

        cv2.putText(img, str(detectscore[0]),
                    (x1, y1 - 5),
                    Config.TEXT_FONT,
                    0.5,
                    Config.TEXT_fontColor,
                    Config.TEXT_lineType)
        # cv2.putText(img, caption,
        #             (loc[0] - 30, loc[1] - 30),
        #             Config.TEXT_FONT,
        #             Config.TEXT_fontScale,
        #             Config.TEXT_fontColor,
        #             Config.TEXT_lineType)
        cv2.putText(img, str(confedece),
                    (x1, y1 - 20),
                    Config.TEXT_FONT,
                    0.5,
                    Config.TEXT_fontColor,
                    Config.TEXT_lineType)
        return img

    @staticmethod
    def DrawROIs(img, locations):
        if(not Config.DEBUG_DRAW_ROIS):
            return img
        imgdraw = img.copy();
        for loc in locations:
            x = loc[0]
            y = loc[1]
            w = loc[2]
            h = loc[3]
            cv2.rectangle(imgdraw, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return imgdraw

    @staticmethod
    def DrawRemoveds(img, locations):
        if (not Config.DEBUG_DRAW_REMOVED_ROIS):
            return img
        imgdraw = img.copy();
        for loc in locations:
            x = loc[0]
            y = loc[1]
            w = loc[2]
            h = loc[3]
            cv2.rectangle(imgdraw, (x, y), (x + w, y + h), (0, 0, 255), 2)
        return imgdraw