"""
author: Xihan Zhao, Louise Lu
datetime: 5/2022
desc: read dataset
"""
from tqdm import tqdm
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import cv2



# CK dataset
class CK(object):

    def __init__(self):
        self.folder = './data/ck+'

    def gen_train(self):

        folder = self.folder

        # CK datast images are restored under different folders thus needs a list called expressions
        expressions = ['anger', 'disgust', 'fear', 'happy', 'neutral','sad', 'surprised']
        x_train = []
        y_train = []
        for i in tqdm(range(len(expressions))):
            expression_folder = os.path.join(folder, expressions[i])
            images = os.listdir(expression_folder)
            for j in range(len(images)):
                img = load_img(os.path.join(expression_folder, images[j]), target_size=(48, 48), color_mode="grayscale")
                img = img_to_array(img)  
                x_train.append(img)
                y_train.append(i)
        x_train = np.array(x_train).astype('float32') / 255.
        y_train = np.array(y_train).astype('int')
        return expressions, x_train, y_train




class RAF(object):
    def __init__(self):
        self.folder = './data/others/aligned'
        self.label_folder = './data/RAF/labels.npy'
    def gen_train(self):
        folder = self.folder
        label_folder = self.label_folder
        images = os.listdir(folder)
        x_train = []
        y= np.load(label_folder)
        y_train = y
        y_train = y-1
        for i in range(len(images)):
                img = load_img(os.path.join(folder, images[i]), target_size=(48, 48), color_mode="grayscale")
                img = img_to_array(img)  
                x_train.append(img)
        x_train = np.array(x_train).astype('float32') / 255.
        y_train = np.array(y_train).astype('int')
        return x_train, y_train
   

    




class Combined(object):
    def __init__(self):
        self.folder = './data/combined'

    def gen_train(self):

        folder = self.folder

        # CK datast images are restored under different folders thus needs a list called expressions
        expressions = ['anger', 'disgust', 'fear', 'happy', 'neutral','sad', 'surprised']
        x_train = []
        y_train = []
        for i in tqdm(range(len(expressions))):
            expression_folder = os.path.join(folder, expressions[i])
            images = os.listdir(expression_folder)
            for j in range(len(images)):
                img = load_img(os.path.join(expression_folder, images[j]), target_size=(48, 48), color_mode="grayscale")
                img = img_to_array(img)  
                x_train.append(img)
                y_train.append(i)
        x_train = np.array(x_train).astype('float32') / 255.
        y_train = np.array(y_train).astype('int')
        return expressions, x_train, y_train