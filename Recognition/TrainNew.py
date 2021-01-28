import time
from shutil import copyfile
import numpy
import tensorflow as tf

from tensorflow.keras.callbacks import TensorBoard
from TrafficSignSystem.Utilities import *

print("Loading Train Datasets...")
train_images ,train_labels = TrafficTrainData()
print("Train Dataset Loaded.")


train_images = numpy.array(train_images)
train_labels = numpy.array(train_labels)

# normalise_images = normalise_images(train_images,train_images)


times = str(time.time()).split('.')[0]
NAME = "Traffic" +  times

start_time = time.time()
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,(3,3), activation="relu",input_shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(43, activation="softmax")
])

tensorboard = TensorBoard(log_dir=Config.LOGS_PATH+"/{}".format(NAME))
model.compile(optimizer="rmsprop",
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=15, batch_size=Config.BATCH_SIZE,callbacks=[tensorboard])
names = (Config.MODELS_PATH+"/x{}"+str(Config.IMAGE_SIZE)+".model").format(NAME)
model.save(names)
print("--- %s seconds ---" % (time.time() - start_time))
copyfile(names, Config.RECOGNITION_MODEL_PATH)