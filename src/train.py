# import system packages
import sys
import csv
import os
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import datetime
import scipy
import matplotlib.pyplot as plt
import json
# import keras and sklearn
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# import local data
from models import GaborCNN3, SimpleCNN3, CNN3
from data import CK, RAF, Jaffe




parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ck+", help="dataset to train, ck+ or raf or jaffe")
parser.add_argument("--model", type=str, default = "gabor_cnn", help="model to train, simple_cnn or gabor_cnn or cnn3 or VGG")
parser.add_argument("--epochs", type=int, default=140)
parser.add_argument("--batch_size", type=int, default=32)

opt = parser.parse_args()
print(opt)


# batchsize = 32
# epoch_size =100



    
tf.debugging.set_log_device_placement(True)
if __name__ == "__main__":
        #######################################################################################################################################################
        if opt.dataset == "ck+" and opt.model == "gabor_cnn":
            with tf.device('/device:GPU:0'):
                expr, x, y = CK().gen_train()

                y = to_categorical(y).reshape(y.shape[0], -1)
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2022)
                x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=2022)
                # x_train, x_valid, x_test, y_train, y_valid, y_test = CK().gen_data()

                train_generator = ImageDataGenerator(rotation_range=10,
                                                width_shift_range=0.05,
                                                height_shift_range=0.05,
                                                horizontal_flip=True,
                                                shear_range=0.1,
                                                zoom_range=0.1
                                                ).flow(x_train, y_train, batch_size=opt.batch_size)
                valid_generator = ImageDataGenerator().flow(x_valid, y_valid, batch_size=opt.batch_size)

                
                model = GaborCNN3(input_shape=(48, 48, 1), n_classes=7)
                sgd = SGD(learning_rate=0.01, decay=1e-4)
                # sgd = SGD(learning_rate=0.01, decay=1e-4, momentum=0.9, nesterov=True)
                #optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

                model.compile(optimizer = sgd, loss='categorical_crossentropy', metrics=['accuracy'])
                
                
                callback = [
                #     EarlyStopping(monitor='val_loss', patience=50, verbose=True),
                #     ReduceLROnPlateau(monitor='lr', factor=0.1, patience=15, verbose=True),
                ModelCheckpoint('./output/ck_cnn3_gaborcnn_weights.h5', monitor='val_acc', verbose=True, save_best_only=True,
                                save_weights_only=True)]

                History = model.fit(train_generator, steps_per_epoch=len(y_train) // opt.batch_size, epochs=opt.epochs,
                                            validation_data=valid_generator, validation_steps=len(y_valid) // opt.batch_size,
                                            callbacks=callback)
                his = History.history
                history_array = np.stack((np.array(his['loss']), np.array(his['val_loss']), np.array(his['accuracy']), np.array(his['val_accuracy'])), axis = -1)
                with open("./output/ck_gaborcnn_history_array",'w') as j:
                    np.savetxt(j, history_array, delimiter=',', header = 'train_loss,val_loss,train_acc,val_acc')
                #json.dump(history_txt, open('./output/history_file.txt', 'w'), cls=PythonObjectEncoder)
                model.save_weights('./output/ck_gaborcnn_weight.h5')
                print("GaborCNN on CK+ dataset training completed!")
        

        #######################################################################################################################################################
        if opt.dataset == "ck+" and opt.model == "simple_cnn":
            with tf.device('/device:GPU:0'):
                expr, x, y = CK().gen_train()
                y = to_categorical(y).reshape(y.shape[0], -1)
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2022)
                x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=2022)

                train_generator = ImageDataGenerator(rotation_range=10,
                                                width_shift_range=0.05,
                                                height_shift_range=0.05,
                                                horizontal_flip=True,
                                                shear_range=0.1,
                                                zoom_range=0.1
                                                ).flow(x_train, y_train, batch_size=opt.batch_size)
                valid_generator = ImageDataGenerator().flow(x_valid, y_valid, batch_size=opt.batch_size)

                
                model = SimpleCNN3(input_shape=(48, 48, 1), n_classes=7)
                sgd = SGD(learning_rate=0.01, decay=1e-4)
                # sgd = SGD(learning_rate=0.01, decay=1e-4, momentum=0.9, nesterov=True)
                #optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

                model.compile(optimizer = sgd, loss='categorical_crossentropy', metrics=['accuracy'])
                
                
                callback = [
                #     EarlyStopping(monitor='val_loss', patience=50, verbose=True),
                #     ReduceLROnPlateau(monitor='lr', factor=0.1, patience=15, verbose=True),
                ModelCheckpoint('./output/ck_simplecnn_weights.h5', monitor='val_acc', verbose=True, save_best_only=True,
                                save_weights_only=True)]

                History = model.fit(train_generator, steps_per_epoch=len(y_train) // opt.batch_size, epochs=opt.epochs,
                                            validation_data=valid_generator, validation_steps=len(y_valid) // opt.batch_size,
                                            callbacks=callback)
                his = History.history
                history_array = np.stack((np.array(his['loss']), np.array(his['val_loss']), np.array(his['accuracy']), np.array(his['val_accuracy'])), axis = -1)
                with open("./output/ck_cnn3_simplecnn_history_array",'w') as j:
                    np.savetxt(j, history_array, delimiter=',', header = 'train_loss,val_loss,train_acc,val_acc')
                #json.dump(history_txt, open('./output/history_file.txt', 'w'), cls=PythonObjectEncoder)
                model.save_weights('./output/ck_simplecnn_weight.h5')
                print("Simple CNN on CK+ dataset training completed!")

        #######################################################################################################################################################
        if opt.dataset == "raf" and opt.model == "gabor_cnn":
            with tf.device('/device:GPU:0'):
                x, y = RAF().gen_train()
                y = to_categorical(y).reshape(y.shape[0], -1)
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2022)
                x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=2022)

                train_generator = ImageDataGenerator(rotation_range=10,
                                                width_shift_range=0.05,
                                                height_shift_range=0.05,
                                                horizontal_flip=True,
                                                shear_range=0.1,
                                                zoom_range=0.1
                                                ).flow(x_train, y_train, batch_size=opt.batch_size)
                valid_generator = ImageDataGenerator().flow(x_valid, y_valid, batch_size=opt.batch_size)

                
                model = GaborCNN3(input_shape=(48, 48, 1), n_classes=7)
                sgd = SGD(learning_rate=0.01, decay=1e-4)
                # sgd = SGD(learning_rate=0.01, decay=1e-4, momentum=0.9, nesterov=True)
                #optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

                model.compile(optimizer = sgd, loss='categorical_crossentropy', metrics=['accuracy'])
                
                
                callback = [
                #     EarlyStopping(monitor='val_loss', patience=50, verbose=True),
                #     ReduceLROnPlateau(monitor='lr', factor=0.1, patience=15, verbose=True),
                ModelCheckpoint('./output/raf_cnn3_gaborcnn_weights.h5', monitor='val_acc', verbose=True, save_best_only=True,
                                save_weights_only=True)]

                History = model.fit(train_generator, steps_per_epoch=len(y_train) // opt.batch_size, epochs=opt.epochs,
                                            validation_data=valid_generator, validation_steps=len(y_valid) // opt.batch_size,
                                            callbacks=callback)
                his = History.history
                history_array = np.stack((np.array(his['loss']), np.array(his['val_loss']), np.array(his['accuracy']), np.array(his['val_accuracy'])), axis = -1)
                with open("./output/raf_gaborcnn_history_array",'w') as j:
                    np.savetxt(j, history_array, delimiter=',', header = 'train_loss,val_loss,train_acc,val_acc')
                #json.dump(history_txt, open('./output/history_file.txt', 'w'), cls=PythonObjectEncoder)
                model.save_weights('./output/raf_gaborcnn_weight.h5')
                print("GaborCNN on RAF dataset training completed!")
        
        #######################################################################################################################################################
        if opt.dataset == "raf" and opt.model == "simple_cnn":
            with tf.device('/device:GPU:0'):
                x, y = RAF().gen_train()
                y = to_categorical(y).reshape(y.shape[0], -1)
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2022)
                x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=2022)

                train_generator = ImageDataGenerator(rotation_range=10,
                                                width_shift_range=0.05,
                                                height_shift_range=0.05,
                                                horizontal_flip=True,
                                                shear_range=0.1,
                                                zoom_range=0.1
                                                ).flow(x_train, y_train, batch_size=opt.batch_size)
                valid_generator = ImageDataGenerator().flow(x_valid, y_valid, batch_size=opt.batch_size)

                
                model = SimpleCNN3(input_shape=(48, 48, 1), n_classes=7)
                sgd = SGD(learning_rate=0.01, decay=1e-4)
                # sgd = SGD(learning_rate=0.01, decay=1e-4, momentum=0.9, nesterov=True)
                #optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

                model.compile(optimizer = sgd, loss='categorical_crossentropy', metrics=['accuracy'])
                
                
                callback = [
                #     EarlyStopping(monitor='val_loss', patience=50, verbose=True),
                #     ReduceLROnPlateau(monitor='lr', factor=0.1, patience=15, verbose=True),
                ModelCheckpoint('./output/raf_cnn3_simplecnn_weights.h5', monitor='val_acc', verbose=True, save_best_only=True,
                                save_weights_only=True)]

                History = model.fit(train_generator, steps_per_epoch=len(y_train) // opt.batch_size, epochs=opt.epochs,
                                            validation_data=valid_generator, validation_steps=len(y_valid) // opt.batch_size,
                                            callbacks=callback)
                his = History.history
                history_array = np.stack((np.array(his['loss']), np.array(his['val_loss']), np.array(his['accuracy']), np.array(his['val_accuracy'])), axis = -1)
                with open("./output/raf_simplecnn_history_array",'w') as j:
                    np.savetxt(j, history_array, delimiter=',', header = 'train_loss,val_loss,train_acc,val_acc')
                #json.dump(history_txt, open('./output/history_file.txt', 'w'), cls=PythonObjectEncoder)
                model.save_weights('./output/raf_simplecnn_weight.h5')
                print("Simple CNN on RAF dataset training completed!")

        #######################################################################################################################################################
        if opt.dataset == "jaffe" and opt.model == "cnn3":
            with tf.device('/device:GPU:0'):
                expr, x, y = Jaffe().gen_train()
                y = to_categorical(y).reshape(y.shape[0], -1)
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2022)
                x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=2022)
                train_generator = ImageDataGenerator(rotation_range=10,
                                                width_shift_range=0.05,
                                                height_shift_range=0.05,
                                                horizontal_flip=True,
                                                shear_range=0.1,
                                                zoom_range=0.1
                                                ).flow(x_train, y_train, batch_size=opt.batch_size)
                valid_generator = ImageDataGenerator().flow(x_valid, y_valid, batch_size=opt.batch_size)

                
                model = CNN3(input_shape=(48, 48, 1), n_classes=7)
                sgd = SGD(learning_rate=0.01)
                # sgd = SGD(learning_rate=0.01, decay=1e-4, momentum=0.9, nesterov=True)
                #optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

                model.compile(optimizer = sgd, loss='categorical_crossentropy', metrics=['accuracy'])
                
                
                callback = [
                #     EarlyStopping(monitor='val_loss', patience=50, verbose=True),
                #     ReduceLROnPlateau(monitor='lr', factor=0.1, patience=15, verbose=True),
                ModelCheckpoint('./output/jaffe_cnn3_weights.h5', monitor='val_acc', verbose=True, save_best_only=True,
                                save_weights_only=True)]

                History = model.fit(train_generator, steps_per_epoch=len(y_train) // opt.batch_size, epochs=opt.epochs,
                                            validation_data=valid_generator, validation_steps=len(y_valid) // opt.batch_size,
                                            callbacks=callback)
                his = History.history
                history_array = np.stack((np.array(his['loss']), np.array(his['val_loss']), np.array(his['accuracy']), np.array(his['val_accuracy'])), axis = -1)
                with open("./output/jaffe_cnn3_history_array",'w') as j:
                    np.savetxt(j, history_array, delimiter=',', header = 'train_loss,val_loss,train_acc,val_acc')
                #json.dump(history_txt, open('./output/history_file.txt', 'w'), cls=PythonObjectEncoder)
                model.save_weights('./output/jaffe_cnn3_weight.h5')
                print("CNN3 on Jaffe dataset training completed!")


        #######################################################################################################################################################
        if opt.dataset == "jaffe" and opt.model == "simple_cnn":
            with tf.device('/device:GPU:0'):
                expr, x, y = Jaffe().gen_train()
                y = to_categorical(y).reshape(y.shape[0], -1)
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2022)
                x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=2022)

                train_generator = ImageDataGenerator(rotation_range=10,
                                                width_shift_range=0.05,
                                                height_shift_range=0.05,
                                                horizontal_flip=True,
                                                shear_range=0.1,
                                                zoom_range=0.1
                                                ).flow(x_train, y_train, batch_size=opt.batch_size)
                valid_generator = ImageDataGenerator().flow(x_valid, y_valid, batch_size=opt.batch_size)

                
                model = SimpleCNN3(input_shape=(48, 48, 1), n_classes=7)
                sgd = SGD(learning_rate=0.01, decay=1e-4)
                # sgd = SGD(learning_rate=0.01, decay=1e-4, momentum=0.9, nesterov=True)
                #optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

                model.compile(optimizer = sgd, loss='categorical_crossentropy', metrics=['accuracy'])
                
                
                callback = [
                #     EarlyStopping(monitor='val_loss', patience=50, verbose=True),
                #     ReduceLROnPlateau(monitor='lr', factor=0.1, patience=15, verbose=True),
                ModelCheckpoint('./output/jaffe_cnn3_simplecnn_weights.h5', monitor='val_acc', verbose=True, save_best_only=True,
                                save_weights_only=True)]

                History = model.fit(train_generator, steps_per_epoch=len(y_train) // opt.batch_size, epochs=opt.epochs,
                                            validation_data=valid_generator, validation_steps=len(y_valid) // opt.batch_size,
                                            callbacks=callback)
                his = History.history
                history_array = np.stack((np.array(his['loss']), np.array(his['val_loss']), np.array(his['accuracy']), np.array(his['val_accuracy'])), axis = -1)
                with open("./output/jaffe_simplecnn_history_array",'w') as j:
                    np.savetxt(j, history_array, delimiter=',', header = 'train_loss,val_loss,train_acc,val_acc')
                #json.dump(history_txt, open('./output/history_file.txt', 'w'), cls=PythonObjectEncoder)
                model.save_weights('./output/jaffe_simplecnn_weight.h5')
                print("Simple CNN on Jaffe dataset training completed!")
