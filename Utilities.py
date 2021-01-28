import cv2
import csv
import pickle
import os
import time
import numpy as np
from time import gmtime, strftime
from Config import Config
import datetime
import shutil
from skimage.feature import hog
from PIL import Image


# load image with grey mode and hsv
def loadImg(imagepath, grey=False, hsv=False):
    img = None
    if (not os.path.exists(imagepath)):
        return img
    img = cv2.imread(imagepath, 1)
    if (grey):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if (hsv):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img


# apply multi masks
def applyMask(image, normaledimage, normaledimage2, maskdict):
    res = None
    returnarray = []
    # open_kern = np.ones((1, 1), dtype=np.uint8)
    img = image.copy()
    imghsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
    normalimghsv = cv2.cvtColor(normaledimage.copy(), cv2.COLOR_BGR2HSV)
    normalimghsv2 = cv2.cvtColor(normaledimage2.copy(), cv2.COLOR_BGR2HSV)
    for key, maskarray in maskdict.items():
        res = None
        for mask in maskarray:
            if mask[2] == 2:
                maskx = cv2.inRange(normalimghsv2.copy(), np.array(mask[0]), np.array(mask[1]))
            elif mask[2] == 1:
                maskx = cv2.inRange(normalimghsv.copy(), np.array(mask[0]), np.array(mask[1]))
            else:
                maskx = cv2.inRange(imghsv.copy(), np.array(mask[0]), np.array(mask[1]))

            # maskx = cv2.morphologyEx(maskx, cv2.MORPH_OPEN, open_kern, iterations=2)

            # imgcopy = img.copy()
            # imgcopy[maskx == 0] = 0
            # imgfinal = imgcopy #cv2.addWeighted(img, .4, imgcopy, .8, 1)

            if res is None:
                res = maskx
            else:
                res = cv2.add(res, maskx)

        # res = cv2.filter2D(res, -1, kernel)
        # res = cv2.bitwise_and(img, img, mask=maskx)
        returnarray.append(res)

    return returnarray


def load_image_for_cnn(filename, startpos, endpos, convertforcnn=True):
    if (convertforcnn):
        # img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(filename)

    img = img[startpos[1]:endpos[1], startpos[0]:endpos[0]]
    if (convertforcnn):
        img = filterimage_for_sign(img)
    return img


def filterArrayimageSings(imgarray):
    array = []
    for img in imgarray:
        if img.shape[0] > 0 and img.shape[1] > 0:
            # cvimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            array.append(filterimage_for_sign(img))
    return np.array(array)


def filterimage_for_sign(img):
    img = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
    # clahe = cv2.createCLAHE(tileGridSize=(3, 3), clipLimit=15.0)
    # img = clahe.apply(img)
    # img = img / 255.0
    return img


def load_image_for_detection(filename):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    return img


def save_array_to_file(array, filename):
    pickle_out = open(filename, "wb")
    pickle.dump(array, pickle_out)
    pickle_out.close()
    return filename


def read_array_from_file(filename):
    pickle_in = open(filename, "rb")
    array = pickle.load(pickle_in)
    return array


def normalise_images(imgs, dist):
    """
    Nornalise the supplied images from data in dist
    """
    std = np.std(dist)
    # std = 128
    mean = np.mean(dist)
    # mean = 128
    return (imgs - mean) / std


def biggestIndex(array):
    id = 0
    max = 0
    for i in range(len(array)):
        if (array[i] > max):
            id = i
            max = array[i]

    return id


def biggest(array):
    max = 0
    for i in range(len(array)):
        if (array[i] > max):
            max = array[i]

    return max


def TrafficTrainData():
    train_images = read_array_from_file(Config.PICKLES_PATH + "train_images.pickle")
    train_labels = read_array_from_file(Config.PICKLES_PATH + "train_labels.pickle")
    return train_images, train_labels


def TrafficTestData():
    test_images = read_array_from_file(Config.PICKLES_PATH + "test_images.pickle")
    test_labels = read_array_from_file(Config.PICKLES_PATH + "test_labels.pickle")
    return test_images, test_labels


def readTrafficSigns(rootpath, convertforcnn=True):
    images = []  # images
    labels = []  # corresponding labelsx
    # loop over all 42 classes Extra One Class
    for c in range(0, 43):
        prefix = rootpath + '/' + format(c, '05d') + '/'  # subdirectory for class
        gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')  # annotations file
        gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
        gtReader.next()  # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            path = prefix + row[0]
            x1 = int(row[3])
            y1 = int(row[4])
            x2 = int(row[5])
            y2 = int(row[6])
            img = load_image_for_cnn(path, (x1, y1), (x2, y2), convertforcnn)
            images.append(img)  # the 1th column is the filename
            labels.append(row[7])  # the 8th column is the label
        gtFile.close()
    return images, labels


def TrafficDetectionData():
    detection_images = read_array_from_file("detection_images.pickle")
    detection_labels = read_array_from_file("detection_labels.pickle")
    return detection_images, detection_labels


def readDetectionDataset(rootpath, fullinfo=False):
    dataset = {}  # images
    gtFile = open(rootpath + Config.DETECT_DATA_SET_GUID_CSV)  # annotations file
    gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
    gtReader.next()  # skip header
    # loop over all images in current annotations file
    current = 0
    for row in gtReader:
        current += 1
        path = row[0]
        # path =rootpath + row[0]
        x1 = int(row[1])
        y1 = int(row[2])
        x2 = int(row[3])
        y2 = int(row[4])
        # row[5]
        # img = load_image_for_detection(path)
        # if(dataset[path] is None):
        #     dataset[path] = [row[5]]
        # else:
        # print (str(current) + "!"+row[5]+"#")
        # if row[5] is None:
        #     dataset[path] = [-1]

        if (not dataset.has_key(path)):
            if (fullinfo):
                dataset[path] = [[int(row[5]), (x1, y1), (x2, y2)]]
            else:
                dataset[path] = [int(row[5])]

        else:
            temp = dataset[path]
            if (fullinfo):
                temp.append([int(row[5]), (x1, y1), (x2, y2)])
            else:
                temp.append(int(row[5]))

            dataset[path] = temp
        #  =  # the 1th column is the filename
        # # images.append(img) # the 1th column is the filename
        # labels.append() # the 8th column is the label
    gtFile.close()
    # return images, labels
    return dataset
    # return paths,labels


def readTestTrafficSigns(rootpath, convertforcnn=True):
    images = []  # images
    labels = []  # corresponding labels
    prefix = rootpath + '/'  # subdirectory for class
    gtFile = open(prefix + 'GT-final_test.csv')  # annotations file
    gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
    gtReader.next()  # skip header
    # loop over all images in current annotations file
    for row in gtReader:
        path = prefix + row[0]
        x1 = int(row[3])
        y1 = int(row[4])
        x2 = int(row[5])
        y2 = int(row[6])
        img = load_image_for_cnn(path, (x1, y1), (x2, y2), convertforcnn)
        images.append(img)  # the 1th column is the filename
        labels.append(row[7])  # the 8th column is the label
    gtFile.close()
    return images, labels


def searchInArray(Array, niddleArray):
    finded = True
    for niddle in niddleArray:
        for item in Array:
            if (item == niddle):
                finded = True
                break
            else:
                finded = False
        if (not finded):
            return False
    return True


def Precent(countall, count):
    return round((float(count) / float(countall)) * 100, 2)


def Log(logname, data, addtime=False):
    filename = Config.LOGS_PATH + logname + ".txt"
    txt = data
    if (addtime):
        txt = ("[" + str(datetime.datetime.now()) + "]#" + data)

    txt = str(txt) + " \n"
    if (os.path.exists(filename)):
        text_file = open(filename, "a")
        text_file.writelines([str(txt)])
        text_file.close()

    else:
        text_file = open(filename, "w")
        text_file.writelines([str(txt)])
        text_file.close()


def DeleteAllFilesInFolder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def ResizeByWidth(img, Width):
    ratio = float(img.shape[0]) / float(img.shape[1])
    newheight = Width * ratio
    return cv2.resize(img, (int(Width), int(newheight)))


def createImageName(FileNumber):
    if (FileNumber < 10):
        return "0000" + str(FileNumber) + ".ppm"
    elif FileNumber < 100:
        return "000" + str(FileNumber) + ".ppm"
    elif FileNumber < 1000:
        return "00" + str(FileNumber) + ".ppm"
    elif FileNumber < 10000:
        return "0" + str(FileNumber) + ".ppm"
    else:
        return str(FileNumber) + ".ppm"


def inRange(value, range):
    return value >= range[0] and value <= range[1]


def MulticvtColor(imgarray, color):
    for i in range(len(imgarray)):
        imgarray[i] = cv2.cvtColor(imgarray[i], color)
    return imgarray


def ShowImgArray(imgarray):
    for i in range(len(imgarray)):
        if (imgarray[i].shape[0] > 0 and imgarray[i].shape[1] > 0):
            cv2.imshow("img" + str(i), imgarray[i])
            # cv2.imwrite("/Users/pouyadark/Desktop/test/img"+ str(i) +".jpg",imgarray[i])


def blockImagesLocations(imgoriginal, arraylocationofintrest):
    img = imgoriginal.copy()
    for arr in arraylocationofintrest:
        pstart = arr[1]
        pend = arr[2]
        img[pstart[1]:pend[1], pstart[0]:pend[0]] = 0
    return img


def WindowBasedImageCrop(imgoriginal):
    img = imgoriginal.copy()
    imgarray = []
    Current_x = 0
    Current_y = 0
    width = img.shape[1]
    height = img.shape[0]
    widthstep = int(width * Config.WINDOW_SLIDE_STEP)
    heightstep = int(height * Config.WINDOW_SLIDE_STEP)
    i = 0
    while (True):
        i += 1
        x1, y1 = Current_x, Current_y
        x2, y2 = int(Current_x + Config.WINDOW_SIZE[0]), int(Current_y + Config.WINDOW_SIZE[1])
        currentimg = img[y1:y2, x1:x2]
        if (currentimg.shape[0] > 0 and currentimg.shape[1] > 0):
            imgarray.append(currentimg)
        if (Current_x + Config.WINDOW_SIZE[0] < width):
            Current_x += widthstep
        elif (Current_y + Config.WINDOW_SIZE[1] < height):
            Current_x = 0
            Current_y = Current_y + heightstep
        else:
            return imgarray


def getImagesLocations(imgoriginal, optionarray):
    img = imgoriginal.copy()
    ar = []
    for arr in optionarray:
        pstart = arr[1]
        pend = arr[2]
        ar.append(img[pstart[1]:pend[1], pstart[0]:pend[0]])
    return ar


def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y: y + windowSize[1], x:x + windowSize[0]])


def loadSignImages():
    if (len(Config.SIGN_IMAGES) == 43):
        return Config.SIGN_IMAGES
    Config.SIGN_IMAGES = []
    for i in range(43):
        img = cv2.imread(Config.SIGNS_PATH + str(i) + ".jpg")
        img = cv2.resize(img, (50, 50))
        Config.SIGN_IMAGES.append(img)
    return Config.SIGN_IMAGES


def saveSigns(arr, Path):
    i = 0
    for img in arr:
        i += 1
        cv2.imwrite(Path + strftime("%Y-%m-%d-%H.%M.%S-" + str(i), gmtime()) + ".jpg", img)


def DetectCorrectSignCount(singClasses, classarray):
    signs = classarray[:]
    c = 0
    detecteds = singClasses[:]
    for i in range(len(signs)):
        for j in range(len(detecteds) - 1, -1, -1):
            if (signs[i] == detecteds[j]):
                c += 1
                del (detecteds[j])
                break
    return c


def FixImageForSVM(img):
    # img = img/2
    # gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    # gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    # gray, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ###clahwe
    # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # lab_planes = cv2.split(lab)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
    # lab_planes[0] = clahe.apply(lab_planes[0])
    # lab = cv2.merge(lab_planes)
    # img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    ###
    img = cv2.resize(img, Config.WINDOW_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # calculate HOG for positive features

    fd = hog(gray, Config.orientations, Config.pixels_per_cell, Config.cells_per_block, block_norm='L2',
             feature_vector=True)  # fd= feature descriptor
    return fd


def getSignType(Classid):
    for type, signs in Config.SIGN_GROUPS.items():
        if (signs.__contains__(Classid)):
            return type
    return "notfound"


def Progressbar(current, max, start_time, correct, partial, wrong):
    if (current % 10 == 0):
        countperpoint = max / Config.PROGRESS_BAR_SIZE
        filled = current / countperpoint
        # unfield = 20 - filled
        chars = ""
        for nn in range(Config.PROGRESS_BAR_SIZE):
            if (nn == Config.PROGRESS_BAR_SIZE / 2):
                chars = chars + str(Precent(max, current)) + "%"
            elif (nn < filled):
                chars = chars + Config.PROGRESS_BAR_FILLED_CHAR
            elif (nn == filled):
                chars = chars + Config.PROGRESS_BAR_CURRENT_CHAR
            else:
                chars = chars + Config.PROGRESS_BAR_UNFILLED_CHAR
        print("[" + chars + "]")
        print("[" + str(current) + "/" +
              str(max) + "] (" +
              str(current / (time.time() - start_time)) + " Test/Second)"
                                                          " Correct: " + str(correct) + " Wrong: " + str(
                    wrong) + " Partial: " + str(partial) + "!")


def PrintResult(count, countofsigns, countofsignsdetected, correct, partial, wrong, start_time, resclasses):
    print("Test Ended!")
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Count Of Images:" + str(count) + " => " + str(Precent(count, count)) + "%")
    print("Count Of Correct:" + str(correct) + " => " + str(Precent(count, correct)) + "%")
    print("Count Of Partial:" + str(partial) + " => " + str(Precent(count, partial)) + "%")
    print("Count Of Wrong :" + str(wrong) + " => " + str(Precent(count, wrong)) + "%")
    print("All Signs :" + str(countofsigns) + " => " + str(Precent(countofsigns, countofsigns)) + "%")
    print("Correct Signs :" + str(countofsignsdetected) + " => " + str(
        Precent(countofsigns, countofsignsdetected)) + "%")
    for i in range(len(resclasses)):
        print(str(i) + " = " + str(resclasses[i][0]) + " / " + str(resclasses[i][1]) + " > " + str(
            Precent(resclasses[i][0], resclasses[i][1])) + " %")


def CountClassess(resclasses, singClasses, classarray):
    resclasses = resclasses.copy()
    for signclass in classarray:
        resclasses[signclass][0] += 1

    for signclass in singClasses:
        resclasses[signclass][1] += 1
    return resclasses
