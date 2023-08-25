import tensorflow as tf
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path


def loadTrainData(gtTrainPath,gtTestPath):
    dfTrain = pd.read_json(gtTrainPath)
    dfTest = pd.read_json(gtTestPath)# reading the groundtruth json file
    print(dfTrain.head())  # printing first five rows of the file
    print(dfTrain.shape)
    print(dfTrain.columns)
    print(dfTest.head())  # printing first five rows of the file
    print(dfTest.shape)
    print(dfTest.columns)

    train_images = []
    for i in tqdm(range(dfTrain.shape[0])):
        #img = image.load_img(dfTrain['imagePath'][i], target_size=(720, 540, 3))
        img = image.load_img(dfTrain['imagePath'][i], color_mode='grayscale')
        img = image.img_to_array(img)
        img = img / 255
        train_images.append(img)
    npImages = np.array(train_images)
    #X_train = (npImages.reshape(dfTrain.shape[0], -1) - np.mean(npImages)) / np.std(npImages)
    X_train = npImages.reshape(dfTrain.shape[0], -1)
    print(X_train.shape, np.mean(X_train), np.std(X_train))

    #Use the code below to normalize the values of y
    columns = dfTrain.columns
    for col in columns:
        if "-x" in col:
            dfTrain[col] = dfTrain[col] / 540
        if "-y" in col:
            dfTrain[col] = dfTrain[col] / 720
        if "numberOfBoxes" in col:
            dfTrain[col] = dfTrain[col] / dfTrain[col].max()

    print(dfTrain.head())
    dfTmp = dfTrain.drop(['imagePath'], axis=1)
    y_train = np.array(dfTmp)
    print(y_train.shape, np.mean(y_train), np.std(y_train))


    test_images = []
    for i in tqdm(range(dfTest.shape[0])):
        # img = image.load_img(dfTest['imagePath'][i], target_size=(720, 540, 3))
        img = image.load_img(dfTest['imagePath'][i], color_mode='grayscale')
        img = image.img_to_array(img)
        img = img / 255
        test_images.append(img)
    npImages = np.array(test_images)
    # X_test = (npImages.reshape(dfTest.shape[0], -1) - np.mean(npImages)) / np.std(npImages)
    X_test = npImages.reshape(dfTest.shape[0], -1)
    print(X_test.shape, np.mean(X_test), np.std(X_test))

    # Use the code below to normalize the values of y
    columns = dfTest.columns
    for col in columns:
        if "-x" in col:
            dfTest[col] = dfTest[col] / 540
        if "-y" in col:
            dfTest[col] = dfTest[col] / 720
        if "numberOfBoxes" in col:
            dfTest[col] = dfTest[col] / dfTest[col].max()

    print(dfTest.head())
    dfTmp = dfTest.drop(['imagePath'], axis=1)
    y_test = np.array(dfTmp)
    print(y_test.shape, np.mean(y_test), np.std(y_test))

    return X_train, X_test, y_train, y_test

def runModelNoCNN(X_train, X_test, y_train, y_test, model=None):
    # Build the model.
    #from keras.optimizers import SGD
    if not model:
        model = Sequential()
        #model.add(Dense(200, input_shape=(720,540,1)))
        model.add(Dense(200, input_dim=X_train.shape[-1]))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(y_train.shape[-1]))
        print(model.summary())

        print("Training the model...")
        #model.compile('adadelta', 'mse')
        model.compile('adam', loss='mse', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), batch_size=32, verbose=2)
    return model

if __name__ == '__main__':
    tf.debugging.set_log_device_placement(True)  # log on which device the operation ran
    # Uncomment this line to run on CPU
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 6.5GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6.3*1024)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
            raise(e)

    trainPath = 'images/otsu-images/train.json'
    testPath = 'images/otsu-images/test.json'
    dataPath = "images/otsu-images/dataframe.json"
    modelPathNoCNN = "images/otsu-images/ThermalImageModelNoCNN.h5"

    X_train, X_test, y_train, y_test = loadTrainData(trainPath,testPath)

    model = runModelNoCNN(X_train, X_test, y_train, y_test)
    print("saving model at %s" % modelPathNoCNN)
    model.save(modelPathNoCNN)
