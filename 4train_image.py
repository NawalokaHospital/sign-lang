import tensorflow as tf
import cv2
import random
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

DATADIR = 'train_images'
CATEGORIES = os.listdir('train_images')
IMG_SIZE = 50
training_data = []
x_train = []
y_train = []
num_of_class = 10


def create_trainig_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for image in os.listdir(path):
            try:
                image_array = cv2.imread(os.path.join(path, image))
                reiszed_image = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
                # plt.imshow(reiszed_image)
                # plt.show()
                training_data.append([reiszed_image, class_num])
                # plt.imshow(reiszed_image)
                # plt.show()
            except Exception as e:
                print("Exception.")


create_trainig_data();
random.shuffle(training_data)

for image, label in training_data:
    l_array = np.zeros(num_of_class)
    l_array[label] = 1
    x_train.append(image)
    y_train.append(l_array)

x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_train = np.array(y_train)
# save the training image set as pickle files
pickle_out = open('train_images_pk', 'wb')
pickle.dump(x_train, pickle_out)
pickle_out.close()

pickle_out = open('train_labels_pk', 'wb')
pickle.dump(y_train, pickle_out)
pickle_out.close()
