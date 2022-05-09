import cv2
import numpy as np
import sys
import os
# import torch
import argparse
# from model_predict import BestModel
# from model1_pred import BestModel
from models import VGG16, GaborCNN3, generate_faces

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import PReLU
#from tensorflow.python.training import checkpoint_utils as cp



def program(input, output):

    # Load the cascade
    face_cascade = cv2.CascadeClassifier('./cascades/data/haarcascade_frontalface_alt.xml')
    # face_cascade = cv2.CascadeClassifier('cascades/third-party/frontalEyes35x16.xml')



    if input =="0":
        # To capture video from webcam. 
        cap = cv2.VideoCapture(0)
    else:
        # To use a video file as input 
        cap = cv2.VideoCapture(input)
    count = 0
    while True:
        if count % 20 == 0:
                # Read the frame
                _, img = cap.read()
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Detect the faces
                faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors=4)
                # Draw the rectangle around each face
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cropped_img = gray[y:y+h,x:x+w]
                    # cropped_img = cv2.resize(cropped_img, (48,48), interpolation = cv2.INTER_AREA)
                    faces = generate_faces(cropped_img)

                    # TO DO
                    weight_path = 'output/ck_gaborcnn_weight.h5'
                    
                    model = GaborCNN3()

                    #model.built = True
                    model.load_weights(weight_path)#,by_name = True, skip_mismatch = True
                    # model.build(input_shape=(1, 48, 48, 1))
                    results = model.predict(faces)
                    result_sum = np.sum(results, axis=0).reshape(-1)
                    label_idx = np.argmax(result_sum, axis=0)
                    #label_dict = {0:'Surprised', 1: 'Fearful', 2: 'Disgusted', 3: 'Happy', 4: 'Sad', 5: 'Angry', 6: 'Neutral' }
                    #label_dict = {0:'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprised', 6: 'Neutral' }
                    label_dict = {0:'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral',5: 'Sad', 6: 'Surprised' }
                    # 'anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'contempt'
                    font = cv2.FONT_HERSHEY_TRIPLEX
                    #text = obj.emotion
                    
                    cv2.putText(img, label_dict[label_idx], (x,y-8), font, 1, (0,255,0), 1)

                    #cv2.imshow('cropped_img', cropped_img)
                # Display
                cv2.imshow('img', img)
                # Stop if escape key is pressed
                k = cv2.waitKey(30) & 0xff
                if k==27:
                    break
    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)




def main():
    try:
        # arguments = get_args(sys.argv)
        # MODE = arguments.get('mode')
        # if MODE == "load":
        #     INPUT = arguments.get('input_dir')
        #     OUTPUT = arguments.get('output_dir')
        #     if INPUT is None : raise TypeError("Input video file directory must be provided.")
        #     if OUTPUT is None : raise TypeError("Output video file directory must be provided.")
        # elif MODE == "camera":
        #     OUTPUT = arguments.get('output_dir')
        #     if OUTPUT is None : raise TypeError("Output video file directory must be provided.")

        parser = argparse.ArgumentParser(description='Facial Expression Recognition Application - JHU Machine Learning CS 601.475/675')
        parser.add_argument("-i", help = "input video file  directory or 0 to activate camera", type=str)
        parser.add_argument("-o", help = "output file (need to be .avi format)", type=str)
        args = parser.parse_args()
    except:
        sys.stderr.write("Invalid arguments\n")
        exit(-1)
    program(args.i,args.o)

if __name__ == "__main__":
    main()