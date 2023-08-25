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
import cv2 as cv

#get_ipython().run_line_magic('matplotlib', 'inline')


def PlotResults(gtDataframe,gtDataPath):
    plotOutPath = os.path.dirname(gtDataPath) + "/result-plots"
    Path(plotOutPath).mkdir(parents=True, exist_ok=True)
    columns = np.array(gtDataframe.columns[2:])
    dfNew = pd.DataFrame(columns= gtDataframe.columns)
    gtDataframe['oldImagePath'] = gtDataframe['imagePath']

    for index, row in gtDataframe.iterrows():
        filename = os.path.splitext(os.path.basename(row.imagePath))[0]
        fileExt = os.path.splitext(os.path.basename(row.imagePath))[1]
        #print(columns)
        numboxes = int((len(columns) / 4))
        print("numboxes = ", numboxes)
        if row.source == 'test':
            img = cv.imread(row.imagePath)
            row.imagePath = plotOutPath + "/" + filename + "_plot" + fileExt
            #print("test row = ", row)
            for box in range(1, numboxes+1):
                tmpColumnX1 = f"box{box}-x1"
                tmpColumnY1 = f"box{box}-y1"
                tmpColumnX2 = f"box{box}-x2"
                tmpColumnY2 = f"box{box}-y2"
                xmin = int(row[tmpColumnX1])
                ymin = int(row[tmpColumnY1])
                xmax = int(row[tmpColumnX2])
                ymax = int(row[tmpColumnY2])
                #print(xmin, ymin, xmax, ymax)
                width = xmax - xmin
                length = ymax - ymin
                area = width * length
                fullarea = img.shape[1] * img.shape[0]
                ratio = area / fullarea
                if ratio > 0.15:
                    print(row.source, " row box = ", filename, xmin, ymin, xmax, ymax)
                    img = cv.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 5)
                    font = cv.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (xmax - 60, ymin + 20)
                    fontScale = 0.5
                    fontColor = (255, 255, 0)
                    lineType = 2
                    img = cv.putText(img, row.source, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
                else:
                    row[tmpColumnX1] = 0
                    row[tmpColumnX2] = 0
                    row[tmpColumnY1] = 0
                    row[tmpColumnY2] = 0

            cv.imwrite(row.imagePath, img)
            dfNew = dfNew.append(row)
            #print("dfNew = ", dfNew.head(5))
        else:
            dfTmp = dfNew[dfNew['oldImagePath'].str.match(row.imagePath)]
            dfTmp.reset_index(inplace=True)
            #print("dfTmp = ", dfTmp)
            newImagePath = dfTmp.loc[0,'imagePath']
            print(newImagePath)
            filename = os.path.splitext(os.path.basename(newImagePath))[0]
            fileExt = os.path.splitext(os.path.basename(newImagePath))[1]
            img = cv.imread(newImagePath)
            #print("pred row = ", row)
            # print(columns)
            for box in range(1, numboxes+1):
                tmpColumnX1 = f"box{box}-x1"
                tmpColumnY1 = f"box{box}-y1"
                tmpColumnX2 = f"box{box}-x2"
                tmpColumnY2 = f"box{box}-y2"
                xmin = int(row[tmpColumnX1])
                ymin = int(row[tmpColumnY1])
                xmax = int(row[tmpColumnX2])
                ymax = int(row[tmpColumnY2])
                #print(xmin, ymin, xmax, ymax)
                width = xmax - xmin
                length = ymax - ymin
                area = width * length
                fullarea = img.shape[1] * img.shape[0]
                ratio = area / fullarea
                if ratio > 0.15:
                    print(row.source, " row box = ", filename, xmin, ymin, xmax, ymax)
                    img = cv.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 5)
                    font = cv.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (xmax - 60, ymin + 20)
                    fontScale = 0.5
                    fontColor = (255, 255, 0)
                    lineType = 2
                    img = cv.putText(img, row.source, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
                else:
                    row[tmpColumnX1] = 0
                    row[tmpColumnX2] = 0
                    row[tmpColumnY1] = 0
                    row[tmpColumnY2] = 0

            cv.imwrite(newImagePath, img)
    return gtDataframe
        #print(dfNew.head(5))



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

def isBoxBig(bbox1,windowW,windowH):
    x1, y1, w1, h1 = bbox1[0], bbox1[2], abs(bbox1[1] - bbox1[0]), abs(bbox1[3] - bbox1[2])
    Window = windowW * windowH

    ratio = (w1*h1) / Window

    if ratio > 0.15:
        return True
    else:
        return False

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

    otsuPath = "images/otsu-mask"
    binPath = "images/binary-mask"
    #basePath = otsuPath
    basePath = binPath

    trainDataPath = basePath + '/train.json'
    testDataPath = basePath + '/test.json'
    modelPath = basePath + "/ThermalImageModelNoBatch.h5"
    infDataPath = 'images/inference/dataframe-binary-inference.json'

    X_test,y_test,dfTest = loadTestData(infDataPath)
    # Uncomment this line in order to evaluate the fit on training data
    #X_test,y_test,dfTest = loadTestData(trainDataPath)

    #evalImages.update(loadImages('data','model_eval/*.webp'))
    # Load images to predict genre for

    # load the model
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
    y_pred[:,:] = ( y_pred[:,:] - y_pred[:,:].min() ) / (y_pred[:,:].max() - y_pred[:,:].min())

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
        ious_row = []
        tmpIou = 0
        for colIndex in range(1, len(columns),4):
            if isBoxBig(y_pred[x,(colIndex):(colIndex + 4)],540,720):
                if isBoxBig(y_test[x, (colIndex):(colIndex+4)], 540, 720):
                    tmpIou = (IOU(y_test[x,(colIndex):(colIndex + 4)],y_pred[x,(colIndex):(colIndex+4)]))
                else:
                    tmpIou = 0
            else:
                if isBoxBig(y_test[x,(colIndex):(colIndex+4)],540,720):
                    tmpIou = (IOU(y_test[x, (colIndex):(colIndex + 4)], y_pred[x, (colIndex):(colIndex + 4)]))
                else:
                    tmpIou = 1
                    y_pred[x, (colIndex):(colIndex + 4)] = 0

            ious_row.append(tmpIou)
        ious.append(sum(ious_row)/len(ious_row))

    print("Average IOU (Metric : Higher is better, max=1) = ", np.array(ious).mean())
    print("IOU length = ", len(ious))
    dfTest['IOU'] = ious
    #print(y_pred)

    dfResult = dfTest[['imagePath','IOU']]
    del dfTest['IOU']
    print(dfResult.head(5))

    dfPred = pd.merge(dfResult,dfYPred,left_index=True, right_index=True)
    print(dfPred.head(5))
    print(dfPred.shape)
    dfPred.to_json(basePath + '/resultsCNN.json')

    dfTestNorm = pd.merge(dfResult,dfYTest,left_index=True, right_index=True)
    print(dfTestNorm.head(5))
    print(dfTestNorm.shape)
    dfTestNorm.to_json(basePath + '/testCNNNormalized.json')

    dfPred['source'] = 'pred'
    dfTestNorm['source'] = 'test'
    dfAppended = pd.concat([dfTestNorm,dfPred], ignore_index=True)
    dfNew = dfAppended.astype({'numberOfBoxes':'int64'})
    df = PlotResults(dfNew, basePath + '/dataframe1.json')
    df.to_json(basePath + '/plotted.json')

    #TODO : in case number of boxes is a mismatch, print as a separate accuracy metric

    #TODO : Add MinRotatedRectangles to predict Rotated images better

    #TODO: Add code to evaluate against unseen inference image files
    # inference image files are in images/inference/dataframe...json 

    #TODO: Find matching predicted box for multi box test case

    #TODO: Augmentations should be done after train and test split
    # COMPLETE
