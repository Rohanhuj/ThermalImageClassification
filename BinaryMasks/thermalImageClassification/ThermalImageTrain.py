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
        print("%s shape " %(dfTrain['imagePath'][i]), img.shape)
    npImages = np.array(train_images)
    #X_train = (npImages.reshape(dfTrain.shape[0], -1) - np.mean(npImages)) / np.std(npImages)
    #X_train = npImages.reshape(dfTrain.shape[0], -1)
    X_train = npImages
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
    #X_test = npImages.reshape(dfTest.shape[0], -1)
    X_test = npImages
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

def runModel(X_train, X_test, y_train, y_test, model=None):
    if not model:
        model = Sequential()
        # Color image
        #model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(720, 540, 3)))
        # Grayscale image
        model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(720, 540, 1)))
        # Linearized image i.e 1D image instead of 2D
        #model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=X_train.shape[1:]))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        # Sigmoid is used for predicting probability e.g. for multi-class classification
        model.add(Dense(y_train.shape[-1], activation='sigmoid'))
        #model.add(Dense(9))
        print(model.summary())

        print("Training the model...")
        # Use the line below for multi-class classification model
        #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        #model.compile('adadelta', 'mse')
        model.compile('adam', loss='mse', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), batch_size=32, verbose=2)
    return model


if __name__ == '__main__':
    tf.debugging.set_log_device_placement(True)  # log on which device the operation ran
    # Uncomment this line to run on CPU
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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

    imagePath = "images"
    otsuPath = imagePath + "/otsu-mask"
    binPath = imagePath + "/binary-mask"
    basePath = binPath
    #basePath = otsuPath

    dataPath = basePath + "/dataframe.json"
    modelPath = basePath + "/ThermalImageModelNoBatch.h5"
    trainPath = basePath + '/train.json'
    testPath =  basePath + '/test.json'

    X_train, X_test, y_train, y_test = loadTrainData(trainPath,testPath)
    model = runModel(X_train, X_test, y_train, y_test)

    print("saving model at %s" % modelPath)
    model.save(modelPath)

    #TODO : for better quantative analysis, plot loss and accuracy curves for training and tests
