from TrafficSignSystem.Utilities import *
import os.path
import os
print("Creating Train Dataset...")
train_images,train_labels = readTrafficSigns(Config.RECOGNITION_TRAIN_DATA_SET_PATH)
print("Compeleted Dataset.")

print("Creating Test Dataset...")
test_images,test_labels = readTestTrafficSigns(Config.RECOGNITION_TEST_DATA_SET_PATH)
print("Compeleted Dataset.")

# Path = Config.POUYA_DATA_SET_PATH
# for i in range(43):
#     for file in os.listdir(Path + Config.PATH_SEPERATOR):
#         if file.endswith(".jpg") or file.endswith(".ppm"):
#             img = load_image_for_cnn(Path + Config.PATH_SEPERATOR + file)
#             train_images.append(img)
#             train_labels.append(i)

print("Saving Dataset...")

save_array_to_file(train_images,Config.PICKLES_PATH + "train_images.pickle")
save_array_to_file(train_labels,Config.PICKLES_PATH + "train_labels.pickle")

save_array_to_file(test_images,Config.PICKLES_PATH + "test_images.pickle")
save_array_to_file(test_labels,Config.PICKLES_PATH + "test_labels.pickle")
print("Dataset Saved.")