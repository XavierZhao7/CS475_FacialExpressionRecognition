"""
author: Xihan Zhao, Louise Lu
datetime: 5/2022
desc: read dataset
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import PReLU
from filter_bank import GaborFilterBank



def GaborCNN3(input_shape=(48, 48, 1), n_classes=7):

    input_layer = Input(shape=input_shape)

    x = GaborFilterBank()(input_layer)


    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = Conv2D(64, (5, 5), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = Conv2D(64, (5, 5), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.75)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.75)(x)
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    return model




def SimpleCNN3(input_shape=(48, 48, 1), n_classes=7):

    input_layer = Input(shape=input_shape)

    x = Conv2D(8, (1, 1), strides=1, padding='same', activation='relu')(input_layer)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = Conv2D(16, (3, 3), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = Conv2D(32, (5, 5), strides=2, padding='same')(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    return model



def VGG16(input_shape=(48, 48, 1), n_classes=7):
    input_layer = Input(shape=input_shape)

    x = GaborFilterBank()(input_layer)
    
    x = Conv2D(64,(3,3),padding="same", activation="relu")(x)
    x = Conv2D(64, (3,3),padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=2)(x)
    x = Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=2)(x)
    x = Conv2D(256, (3,3), padding="same", activation="relu")(x)
    x = Conv2D(256, (3,3), padding="same", activation="relu")(x)
    x = Conv2D(256, (3,3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=2)(x)
    x = Conv2D(512, (3,3), padding="same", activation="relu")(x)
    x = Conv2D(512, (3,3), padding="same", activation="relu")(x)
    x = Conv2D(512, (3,3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=2)(x)
    x = Conv2D(512, (3,3), padding="same", activation="relu")(x)
    x = Conv2D(512, (3,3), padding="same", activation="relu")(x)
    x = Conv2D(512, (3,3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=2)(x)
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    return model


def generate_faces(face_img, img_size=48):
    import cv2
    import numpy as np
    face_img = cv2.resize(face_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    resized_images = list()
    resized_images.append(face_img[:, :])
    resized_images.append(face_img[2:45, :])
    resized_images.append(cv2.flip(face_img[:, :], 1))
    resized_images.append(face_img[0:45, 0:45])
    resized_images.append(face_img[2:47, 0:45])
    resized_images.append(face_img[2:47, 2:47])

    for i in range(len(resized_images)):
        resized_images[i] = cv2.resize(resized_images[i], (img_size, img_size))
        resized_images[i] = np.expand_dims(resized_images[i], axis=-1)
    resized_images = np.array(resized_images)
    return resized_images