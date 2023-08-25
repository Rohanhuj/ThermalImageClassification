import os
from pathlib import Path
import tensorflow as tf
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
#get_ipython().run_line_magic('matplotlib', 'inline')

def loadTestData(gtTestPath):
    dfTest = pd.read_json(gtTestPath)  # reading the groundtruth json file
    print(dfTest.head())  # printing first five rows of the file
    print(dfTest.shape)
    print(dfTest.columns)

    test_images = []
    for i in tqdm(range(dfTest.shape[0])):
        # img = image.load_img(dfTest['imagePath'][i], target_size=(720, 540, 3))
        img = image.load_img(dfTest['imagePath'][i], color_mode='grayscale')
        img = image.img_to_array(img)
        img = img / 255
        test_images.append(img)
    npImages = np.array(test_images)
    #X_test = (npImages.reshape(dfTest.shape[0], -1) - np.mean(npImages)) / np.std(npImages)
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

    return X_test, y_test, dfTest

def IOU(bbox1, bbox2):
    '''Calculate overlap between two bounding boxes [x, y, w, h] as the area of intersection over the area of unity'''
    x1, y1, w1, h1 = bbox1[0], bbox1[2],abs(bbox1[1] - bbox1[0]),abs(bbox1[3] - bbox1[2])
    x2, y2, w2, h2 = bbox2[0], bbox2[2], abs(bbox2[1] - bbox2[0]), abs(bbox2[3] - bbox2[2])

    w_I = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_I = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_I <= 0 or h_I <= 0:  # no overlap
        return 0.
    I = w_I * h_I

    U = w1 * h1 + w2 * h2 - I

    return I / U



if __name__ == '__main__':
    tf.debugging.set_log_device_placement(True)  # log on which device the operation ran
    # Uncomment this line to run on CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 6.5GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6.3 * 1024)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
            raise (e)

    X_test,y_test,dfTest = loadTestData('images/otsu-images/test.json')
    # Uncomment this line in order to evaluate the fit on training data
    #X_test,y_test,dfTest = loadTestData('images/otsu-images/train.json')

    #evalImages.update(loadImages('data','model_eval/*.webp'))
    # Load images to predict genre for

    # load the model
    modelPath = "images/otsu-images/ThermalImageModelNoCNN.h5"

    model = load_model(modelPath)
    print(model.summary())

    # Load training data frame for class(column) names i.e. genres
    columns = np.array(dfTest.columns[1:])
    print(len(columns))

    # Predict bounding boxes on the test images.
    y_pred = model.predict(X_test)
    print(y_test.shape)
    print(y_pred.shape)
    print(y_test[1,1:5])
    print(y_pred[1,1:5])

    #Calculate measured accuracy of model using IOU[intersection over union]
    #compute IOU for y_test vs y_pred for each corresponding bounding box
    #print out accuracy for each imagePath
    #finally print out average accuracy across all test images

    # Scale the predictions since NN will only predict between 0 and 1, and numberOfBoxes data could throw it off...
    y_pred[:,1:] = ( y_pred[:,1:] - y_pred[:,1:].min() ) / (y_pred[:,1:].max() - y_pred[:,1:].min())

    # Use the code below to denormalize the values of y
    for colIndex in range(0, len(columns)):
        if "-x" in columns[colIndex]:
            y_pred[:,colIndex] *= 540
            y_test[:,colIndex] *= 540
        if "-y" in columns[colIndex]:
            y_pred[:,colIndex] *= 720
            y_test[:,colIndex] *= 720
        if "numberOfBoxes" in columns[colIndex]:
            y_pred[:, colIndex] *= int(len(columns)/4)
            y_test[:, colIndex] *= int(len(columns)/4)

    dfYPred = pd.DataFrame(y_pred,columns=columns)
    print(dfYPred.head(5))
    print(dfYPred.shape)

    dfYTest = pd.DataFrame(y_test,columns=columns)
    print(dfYTest.head(5))
    print(dfYTest.shape)

    #print(IOU(y_test[1,1:5],y_pred[1,1:5]))
    #print(dfResult.loc[1])
    print(dfTest.loc[1,'imagePath'],IOU(y_test[1,1:5],y_pred[1,1:5]))
    ious = []
    for x in range(0,dfTest.shape[0]):
        ious.append(IOU(y_test[x,1:5],y_pred[x,1:5]))
        #TODO : Add support for computing IOU for second box also
        #TODO : If area of detected box is less than 0.15 of full
        #image (see imagePrep.py) then default to zeroes and then compare
    print("Average IOU (Metric : Higher is better, max=1) = ", np.array(ious).mean())
    dfTest['IOU'] = ious
    #print(y_pred)

    dfResult = dfTest[['imagePath','IOU']]
    del dfTest['IOU']
    print(dfResult.head(5))

    dfPred = pd.merge(dfResult,dfYPred,left_index=True, right_index=True)
    print(dfPred.head(5))
    print(dfPred.shape)
    dfPred.to_json(r'images/otsu-images/resultsNoCNN.json')

    dfTestNorm = pd.merge(dfResult,dfYTest,left_index=True, right_index=True)
    print(dfTestNorm.head(5))
    print(dfTestNorm.shape)
    dfTestNorm.to_json(r'images/otsu-images/testNoCNNNormalized.json')

    #TODO : in case number of boxes is a mismatch, print as a separate accuracy metric
