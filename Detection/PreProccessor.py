
import cv2
import Config
#Pre Proccess
from Utilities import *

class PreProccessor:

    #thres
    @staticmethod
    def Normalize2(img):
        rgb = img.copy()
        norm = np.zeros(img.shape, np.float32)
        norm_rgb = np.zeros(img.shape, np.uint8)

        b = rgb[:, :, 0]
        g = rgb[:, :, 1]
        r = rgb[:, :, 2]

        sum = np.sqrt(
            np.multiply(b.astype(int), b.astype(int))
            + np.multiply(g.astype(int), g.astype(int))
            + np.multiply(r.astype(int), r.astype(int))
        )
        # sum = np.int(b) + np.int(g) + np.int(r)

        norm[:, :, 0] = b.astype(int) / sum.astype(float) * 255.0
        norm[:, :, 1] = g.astype(int) / sum.astype(float) * 255.0
        norm[:, :, 2] = r.astype(int) / sum.astype(float) * 255.0

        norm_rgb = cv2.convertScaleAbs(norm)
        return norm_rgb
    @staticmethod
    def Normalize(img):

        # lookUpTable = np.empty((1, 256), np.uint8)
        # for i in range(256):
        #     lookUpTable[0, i] = np.clip(pow(i / 255.0, 0.8) * 255.0, 0, 255)
        # res = cv2.LUT(img.copy(), lookUpTable)


        # rgb = img.copy()
        # norm = np.zeros(img.shape, np.float32)
        # norm_rgb = np.zeros(img.shape, np.uint8)
        #
        # b = rgb[:, :, 0]
        # g = rgb[:, :, 1]
        # r = rgb[:, :, 2]
        #
        # sum = np.sqrt(np.multiply(b, b)) + np.sqrt(np.multiply(g, g)) +  np.sqrt(np.multiply(r, r))
        #
        # norm[:, :, 0] = b / sum * 255.0
        # norm[:, :, 1] = g / sum * 255.0
        # norm[:, :, 2] = r / sum * 255.0
        #
        # norm_rgb = cv2.convertScaleAbs(norm)
        # return norm_rgb

        rgb = img.copy()
        norm = np.zeros(img.shape, np.float32)
        norm_rgb = np.zeros(img.shape, np.uint8)

        b = rgb[:, :, 0]
        g = rgb[:, :, 1]
        r = rgb[:, :, 2]

        sum = b.astype(int) + g.astype(int) + r.astype(int)
        # sum = np.int(b) + np.int(g) + np.int(r)

        norm[:, :, 0] = b.astype(int) / sum.astype(float) * 255.0
        norm[:, :, 1] = g.astype(int) / sum.astype(float) * 255.0
        norm[:, :, 2] = r.astype(int) / sum.astype(float) * 255.0

        norm_rgb = cv2.convertScaleAbs(norm)


        # rgb = img.copy()
        # norm = np.zeros(img.shape, np.float32)
        # norm_rgb = np.zeros(img.shape, np.uint8)
        #
        # b = rgb[:, :, 0]
        # g = rgb[:, :, 1]
        # r = rgb[:, :, 2]
        #
        # sum = np.sqrt(
        #     np.multiply(b.astype(int), b.astype(int))
        #     + np.multiply(g.astype(int), g.astype(int))
        #     + np.multiply(r.astype(int), r.astype(int))
        # )
        # # sum = np.int(b) + np.int(g) + np.int(r)
        #
        # norm[:, :, 0] = b.astype(int) / sum.astype(float) * 255.0
        # norm[:, :, 1] = g.astype(int) / sum.astype(float) * 255.0
        # norm[:, :, 2] = r.astype(int) / sum.astype(float) * 255.0
        #
        # norm_rgb = cv2.convertScaleAbs(norm)
        return norm_rgb
        # return res

    @staticmethod
    def Thresholding(imgarray):
        retarray = []
        for img in imgarray:
            ret, thresh = cv2.threshold(img, Config.THRESHOLD_MIN, Config.THRESHOLD_MAX, cv2.THRESH_BINARY)
            kernel = np.ones((2, 2), np.uint8)
            img = cv2.dilate(thresh, kernel, iterations=1)
            retarray.append(img)
        return retarray

    #Blur image
    @staticmethod
    def BlurImage(image):
        img = image.copy()
        img = cv2.GaussianBlur(img, (3,3), 2)
        return img

    # Sobel image
    @staticmethod
    def Sobel(image):
        img = image.copy()
        img = np.float32(img) / 255.0
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        return mag
    #Clahe
    @staticmethod
    def Clahe(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(Config.CLAHE_GRID_SIZE, Config.CLAHE_GRID_SIZE))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return img

    #remove small object
    @staticmethod
    def RemoveSmallObjects(img):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
        # connectedComponentswithStats yields every seperated component with information on each of them, such as size
        # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1]
        width = stats[1:,cv2.CC_STAT_WIDTH]
        height = stats[1:,cv2.CC_STAT_HEIGHT]
        nb_components = nb_components - 1

        # minimum size of particles we want to keep (number of pixels)
        # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever

        # your answer image
        # img2 = np.zeros((output.shape))
        # for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):

            if sizes[i] >= Config.MIN_OBJECT_SIZE:
                ratio = float(width[i]) / float(height[i])
                if(ratio >= Config.MIN_ROI_RATIO and ratio <= Config.MAX_ROI_RATIO):

                    img[output == i + 1] = 255
                else:
                    img[output == i + 1] = 0
            else:
                img[output == i + 1] = 0
        return img
