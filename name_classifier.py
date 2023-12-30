
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from keras.utils import to_categorical
import random

dic = {0 : 'thumb', 1 : 'index', 2 : 'middle', 3 : 'ring', 4 : 'little'}

def extract_label(img_path,train = True):
    filename,_ = os.path.splitext(os.path.basename(img_path))
    subject_id, etc = filename.split('__')


    gender, lr, finger = etc.split("_")

    print(finger)
    gender = 0 if gender == 'R' else 1
    lr = 0 if lr == 'L' else 1
    if finger == 'thumb':
      finger = 0
    elif finger == 'index':
      finger = 1
    elif finger == 'middle':
      finger = 2
    elif finger == 'ring':
      finger = 3
    elif finger == 'little':
      finger = 4
    return np.array([finger], dtype=np.uint16)



def loading_data(path,boolean):
    data = []
    
    img_size = 96 
    data = []

    img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(img_array, (img_size, img_size))
    label = extract_label(path,boolean)

    data.append([label[0], img_resize ])
    return data



def get_features():
    DATA = "Path to dataset"
    img_size = 96

    data = loading_data(DATA,True)
    random.shuffle(data)

    img, labels = [], []
    for label, feature in data:
        labels.append(label)
        img.append(feature)
    train_data = np.array(img).reshape(-1, img_size, img_size, 1)
    train_data = train_data / 255.0

    train_labels = to_categorical(labels, num_classes = 5)

    return train_data , train_labels



def cnn ():
    model = Sequential([

    Conv2D(32, 3, padding='same', activation='relu',kernel_initializer='he_uniform', input_shape = [96, 96, 1]),
    MaxPooling2D(2),
    Conv2D(64, 3, padding='same', kernel_initializer='he_uniform', activation='relu'),
    MaxPooling2D(2),
    # Conv2D(32, 3, padding='same', kernel_initializer='he_uniform', activation='relu'),
    # MaxPooling2D(2),
    Conv2D(64, 3, padding='same', kernel_initializer='he_uniform', activation='relu'),
    MaxPooling2D(2),
    Flatten(),
    Dense(128,activation = 'relu'),
    Dense(5, activation ='softmax'),
    ])


    model.compile(optimizer = optimizers.Adam(1e-4), loss=tf.keras.losses.CategoricalCrossentropy(), metrics = ['accuracy'])
    model.load_weights('NameFP2.h5')
    return model


def Name_prediction(path):
    model=cnn()
    test_data = loading_data(path,True)

    x_test,y_test= [], []
    for label, feature in test_data:
        y_test.append(label)
        x_test.append(feature)

    x_test = np.array(x_test).reshape(-1, 96, 96, 1)
    pred = ((model.predict(x_test)>0.5).astype(np.int32))
    pred=np.argmax(pred, axis=1)

    return dic[pred[0]]

