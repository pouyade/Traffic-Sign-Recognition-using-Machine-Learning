import Utilities
from Config import Config
import cv2
import os
import time
start_time = time.time()
print("Creating Detect Dataset...")
train_images,train_labels = Utilities.readTrafficSigns(Config.RECOGNITION_TRAIN_DATA_SET_PATH,False)
# test_images,test_labels = Utilities.readTestTrafficSigns(Config.RECOGNITION_TEST_DATA_SET_PATH)

detection_dataset = Utilities.readDetectionDataset(Config.DETECT_TRAIN_DATA_SET_PATH,True)
negetive_images = []
postive_images = train_images
for filename, optionarray in detection_dataset.items():
    img = cv2.imread(Config.DETECT_TRAIN_DATA_SET_PATH + filename)
    postive_images.extend(Utilities.getImagesLocations(img,optionarray))
    img = Utilities.blockImagesLocations(img,optionarray)
    img = Utilities.WindowBasedImageCrop(img)
    negetive_images.extend(img)

path_postive =Config.DATA_SET_PATH + "DetectionDataset/1/";
path_Negetive =Config.DATA_SET_PATH + "DetectionDataset/0/";
for file in os.listdir(path_postive):
    if file.endswith(".jpg"):
        img = cv2.imread(path_postive + file)
        postive_images.append(img)
for file in os.listdir(path_Negetive):
    if file.endswith(".jpg"):
        img = cv2.imread(path_Negetive + file)
        negetive_images.append(img)

Utilities.save_array_to_file(postive_images,Config.PICKLES_PATH + "detection_train_postive_images.pickle")
Utilities.save_array_to_file(negetive_images,Config.PICKLES_PATH + "detection_train_negetive_images.pickle")
# i = 0
# for img in negetive_images:
#     name = "xx" + str(i)
#     i+=1
#     cv2.imwrite("/Users/pouyadark/Documents/ML/opencv-haar-classifier-training-master/negative_images/" + name + ".jpg",img)
#
# i = 0
# for img in postive_images:
#     name = "pp" + str(i)
#     i+=1
#     cv2.imwrite("/Users/pouyadark/Documents/ML/opencv-haar-classifier-training-master/positive_images/" + name + ".jpg",img)

print("Dataset Saved.")
print("--- %s seconds ---" % (time.time() - start_time))