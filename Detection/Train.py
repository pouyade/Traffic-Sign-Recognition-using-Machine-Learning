import Utilities
from Config import Config


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import cv2
import time
start_time = time.time()
print("Start Loading Dataset...")
postive_imgs = Utilities.read_array_from_file(Config.PICKLES_PATH + "detection_train_postive_images.pickle")
negetive_imgs = Utilities.read_array_from_file(Config.PICKLES_PATH + "detection_train_negetive_images.pickle")
print("Loaded Dataset!")

data= []
labels = []

for img in postive_imgs:  # this loop enables reading the files in the pos_im_listing variable one by one

    fd = Utilities.FixImageForSVM(img)
    data.append(fd)
    labels.append(1)

# Same for the negative images
for img in negetive_imgs:
    fd = Utilities.FixImageForSVM(img)
    data.append(fd)
    labels.append(0)

le = LabelEncoder()
labels = le.fit_transform(labels)

#Model
print("->Constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
	np.array(data), labels, test_size=0.0005, random_state=42)
#%% Train the linear SVM
print("->Training Linear SVM classifier...")
model = LinearSVC()
# print(trainData)
# print(trainLabels)
model.max_iter = 4000
model.fit(np.array(trainData), trainLabels)
#%% Evaluate the classifier
print("->Evaluating classifier on test data ...")
predictions = model.predict(np.array(data))
print(classification_report(labels , predictions))

# print accuracy_score(labels, predictions)
# Save the model:
#%% Save the Model
joblib.dump(model, Config.DETECTION_MODEL_PATH)

print("--- %s seconds ---" % (time.time() - start_time))
