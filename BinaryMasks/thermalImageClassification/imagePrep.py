import cv2 as cv
import os
from pathlib import Path
import sys
import pandas as pd
import numpy

# import shutil
# import argparse

# from matplotlib import pyplot as plt

from keras.preprocessing.image import ImageDataGenerator

# Default columns but more are added per box
from sklearn.model_selection import train_test_split



def resize(image, scale_percent=50):
    # scale_percent = 50  # percent of original size
    #width = int(image.shape[1] * scale_percent / 100)
    #height = int(image.shape[0] * scale_percent / 100)
    width = 540
    height = 720
    dim = (width, height)
    # resize image
    resized = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    return resized


def augment(augPath, augPrefix, augtype, X_train, Y_train=None):

    X_train = X_train.reshape((1, X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    # X_train = X_train.astype('float32')

    # Initialize all augmentation type flags to None
    shift = 0
    rotation = 0  # In degrees
    flip = False
    standardize = False
    whiten = False
    augPrefix1 = augPrefix
    augPrefix2 = augPrefix
    BatchSize = 1
    MaxBatchesPerType = 5
    batches = 0

    if 'Shift' in augtype:
        shift = 0.2
        augPrefix1 += "_shift"
        batches += MaxBatchesPerType
    if 'Rotate' in augtype:
        rotation = 90
        augPrefix1 += "_rotate"
        batches += MaxBatchesPerType
    if 'Flip' in augtype:
        flip = True
        augPrefix1 += "_flip"
        batches += MaxBatchesPerType
    if 'Standardize' in augtype:
        standardize = True
        augPrefix2 += "_standardize"
        batches += MaxBatchesPerType
    if 'Whiten' in augtype:  # Whiten not supported yet as it needs too much memory for 540x720 images
        # whiten = True
        # augPrefix2 += "_whiten"
        # batches += MaxBatchesPerType
        pass

    count = 0
    if shift or flip or rotation:
        datagen = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift, rotation_range=rotation,
                                     horizontal_flip=flip, vertical_flip=flip, dtype=int)
        datagen.fit(X_train)
        if Y_train:
            for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=BatchSize, save_to_dir=augPath,
                                                 save_prefix=augPrefix1, save_format='jpg'):
                count += 1
                if count >= batches:
                    break
        else:
            for X_batch in datagen.flow(X_train, Y_train, batch_size=BatchSize, save_to_dir=augPath,
                                        save_prefix=augPrefix1, save_format='jpg'):
                count += 1
                if count >= batches:
                    break
    count = 0
    if whiten or standardize:  # Keep whiten and standardize as separate option as they cannot handle dtype=int
        datagen = ImageDataGenerator(zca_whitening=whiten, featurewise_center=standardize,
                                     featurewise_std_normalization=standardize)
        datagen.fit(X_train)
        if Y_train:
            for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=BatchSize, save_to_dir=augPath,
                                                 save_prefix=augPrefix2, save_format='jpg'):
                count += 1
                if count >= batches:
                    break
        else:
            for X_batch in datagen.flow(X_train, Y_train, batch_size=BatchSize, save_to_dir=augPath,
                                        save_prefix=augPrefix2, save_format='jpg'):
                count += 1
                if count >= batches:
                    break

def dataSplit(gtPath):
    df = pd.read_json(gtPath) # ground truth
    dfTrain, dfTest = train_test_split(df, test_size=0.2)
    print(len(dfTrain))
    print(len(dfTest))

    # Create destination base dir to store preprocessed features and data if it does not exist
    destPath = os.path.dirname(gtPath)
    trainPath = destPath + '/train.json'
    testPath = destPath + '/test.json'

    # Save datasets for training and test
    dfTrain.to_json(trainPath,orient='records')
    dfTest.to_json(testPath,orient='records')

    return trainPath, testPath

def BoxImages(otsuendpath, binaryendpath, otsuboxendpath, binboxendpath, otsuGraypath, binaryGraypath,
              img, filename,negative):

    if negative:
        print("Generating Masked Images from " + filename)
        row = {}


        row["box1-x1"] = 0
        row["box1-x2"] = 0
        row["box1-y1"] = 0
        row["box1-y2"] = 0
        row["numberOfBoxes"] = 0

        print(row)
        return row, row
    else:
        print("Generating Masked Images from " + filename)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_color = img

        # out of 255
        binary_mask_threshold = 125

        # add blur to reduce noise
        blur = cv.GaussianBlur(img_gray, (11, 11), cv.BORDER_CONSTANT)
        blur2 = cv.GaussianBlur(blur, (7, 7), cv.BORDER_CONSTANT)

        # using binary and otsu mask
        ret1, binary_mask = cv.threshold(blur2, binary_mask_threshold, 255, cv.THRESH_BINARY)
        #ret_otsu, otsu_mask = cv.threshold(blur2, 0, 255, cv.THRESH_OTSU)
        ret_otsu, otsu_mask = cv.threshold(blur2, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        gray_with_binary_mask = cv.bitwise_and(img_gray, binary_mask)
        cv.imwrite(binaryGraypath + '/' + filename + '_bingray.jpg', gray_with_binary_mask)

        gray_with_otsu_mask = cv.bitwise_and(img_gray, otsu_mask)
        cv.imwrite(otsuGraypath + '/' + filename + '_otusgray.jpg', gray_with_otsu_mask)

        # converting mask to 3 channels to put onto color image
        bin_mask_three_channels = cv.cvtColor(binary_mask, cv.COLOR_GRAY2BGR)
        otsu_mask_three_channels = cv.cvtColor(otsu_mask, cv.COLOR_GRAY2BGR)

        color_with_binary_mask = cv.bitwise_and(img_color, bin_mask_three_channels)
        cv.imwrite(binaryendpath + '/' + filename + '_bin.jpg', color_with_binary_mask)

        color_with_otsu_mask = cv.bitwise_and(img_color, otsu_mask_three_channels)
        cv.imwrite(otsuendpath + '/' + filename + '_otsu.jpg', color_with_otsu_mask)

        # add boxes to dataframe row
        print("Creating new row")

        rowBin = {}
        validBoxes = 0

        ret_bin, components_bin = cv.connectedComponents(gray_with_binary_mask.astype(numpy.uint8))
        ret_otsu, components_otsu = cv.connectedComponents(gray_with_otsu_mask.astype(numpy.uint8))
        print(f"Number of boxes found using otsu mask : {ret_otsu}")
        print(f"Number of boxes found using binary mask : {ret_bin}")
        numpy.set_printoptions(threshold=sys.maxsize)

        binimageWithBoxes = numpy.zeros((components_bin.shape[0], components_bin.shape[1], 3), dtype=numpy.uint8)

        for f in range(1, components_bin.max() + 1):
            d = numpy.random.randint(0, 255, 3)
            cmy, cmx = numpy.nonzero(components_bin == f)
            cx1 = max(cmx.min() - 2, 0)
            cx2 = min(cmx.max() + 2, binimageWithBoxes.shape[1] - 1)
            cy1 = max(cmy.min() - 2, 0)
            cy2 = min(cmy.max() + 2, binimageWithBoxes.shape[0] - 1)

            width2 = cx2 - cx1
            length2 = cy2 - cy1
            area2 = width2 * length2
            fullarea2 = binimageWithBoxes.shape[1] * binimageWithBoxes.shape[0]
            ratio2 = area2 / fullarea2
            # Use boxes greater than 8% in size relative to image size
            if ratio2 > 0.10:
                binimageWithBoxes[components_bin == f] = d
                binimageWithBoxes[cy1, cx1:cx2, :] = d
                binimageWithBoxes[cy2, cx1:cx2, :] = d
                binimageWithBoxes[cy1:cy2, cx1, :] = d
                binimageWithBoxes[cy1:cy2, cx2, :] = d

                validBoxes += 1
                tempColumnName = f"box{validBoxes}-x1"
                rowBin[tempColumnName] = cx1
                tempColumnName = f"box{validBoxes}-x2"
                rowBin[tempColumnName] = cx2
                tempColumnName = f"box{validBoxes}-y1"
                rowBin[tempColumnName] = cy1
                tempColumnName = f"box{validBoxes}-y2"
                rowBin[tempColumnName] = cy2

        rowBin["numberOfBoxes"] = validBoxes
        print("Binary masked dateframe row = ", rowBin)
        cv.imwrite(binboxendpath + '/' + filename + '_binbox.jpg', binimageWithBoxes)

        imageWithBoxes = numpy.zeros((components_otsu.shape[0], components_otsu.shape[1], 3), dtype=numpy.uint8)
        rowOtsu = {}
        validBoxes = 0
        for i in range(1, components_otsu.max() + 1):
            c = numpy.random.randint(0, 255, 3)
            my, mx = numpy.nonzero(components_otsu == i)
            x1 = max(mx.min() - 2, 0)
            x2 = min(mx.max() + 2, imageWithBoxes.shape[1] - 1)
            y1 = max(my.min() - 2, 0)
            y2 = min(my.max() + 2, imageWithBoxes.shape[0] - 1)

            width = x2 - x1
            length = y2 - y1
            area = width * length
            fullarea = imageWithBoxes.shape[1] * imageWithBoxes.shape[0]
            ratio = area / fullarea
            # Use boxes greater than 8% in size relative to image size
            if ratio > 0.12:
                imageWithBoxes[components_otsu == i] = c
                imageWithBoxes[y1, x1:x2, :] = c
                imageWithBoxes[y2, x1:x2, :] = c
                imageWithBoxes[y1:y2, x1, :] = c
                imageWithBoxes[y1:y2, x2, :] = c

                validBoxes += 1
                tempColumnName = f"box{validBoxes}-x1"
                rowOtsu[tempColumnName] = x1
                tempColumnName = f"box{validBoxes}-x2"
                rowOtsu[tempColumnName] = x2
                tempColumnName = f"box{validBoxes}-y1"
                rowOtsu[tempColumnName] = y1
                tempColumnName = f"box{validBoxes}-y2"
                rowOtsu[tempColumnName] = y2

        rowOtsu["numberOfBoxes"] = validBoxes
        print("Otsu masked dataframe row = ", rowOtsu)
        cv.imwrite(otsuboxendpath + '/' + filename + '_otsubox.jpg', imageWithBoxes)
        return rowOtsu, rowBin


# Main ...

print("Creating Dataframes")
columns = [
    "imagePath", "numberOfBoxes", \
    "box1-x1", "box1-x2", "box1-y1", "box1-y2"
]

dataOtsu = pd.DataFrame(columns=columns)
dataBin = pd.DataFrame(columns=columns)

print("Creating path names")
# Input Paths
imagePath = 'images'
negativePath = imagePath + '/negative'
regularPath = imagePath + '/regular'

# Output Paths
binPath = imagePath + '/binary-mask'
binGrayPath = binPath + '/binary-gray'
binBoxPath = binPath + '/boxed-images'
binMaskedPath = binPath + '/masked-images'

otsuPath = imagePath + '/otsu-mask'
otsuGrayPath = otsuPath + '/otsu-gray'
otsuBoxPath = otsuPath + '/boxed-images'
otsuMaskedPath = otsuPath + '/masked-images'

augPath = imagePath + '/augment'
augBoxOtsuPath = otsuPath + '/augment/boxed-images'
augBoxBinPath = binPath + '/augment/boxed-images'
inferencePath = imagePath + '/inference'

Path(binPath).mkdir(parents=True, exist_ok=True)
Path(otsuPath).mkdir(parents=True, exist_ok=True)
Path(augPath).mkdir(parents=True, exist_ok=True)
Path(otsuGrayPath).mkdir(parents=True, exist_ok=True)
Path(binGrayPath).mkdir(parents=True, exist_ok=True)
Path(otsuBoxPath).mkdir(parents=True, exist_ok=True)
Path(binBoxPath).mkdir(parents=True, exist_ok=True)
Path(otsuMaskedPath).mkdir(parents=True, exist_ok=True)
Path(binMaskedPath).mkdir(parents=True, exist_ok=True)
Path(augBoxOtsuPath).mkdir(parents=True, exist_ok=True)
Path(augBoxBinPath).mkdir(parents=True, exist_ok=True)
Path(regularPath).mkdir(parents=True, exist_ok=True)

dataOtsuPath = otsuPath + '/dataframe-otsu.json'
dataBinPath = binPath + '/dataframe-binary.json'
dataOtsuInferencePath = inferencePath + '/dataframe-otsu-inference.json'
dataBinInferencePath = inferencePath + '/dataframe-binary-inference.json'

print("Generating bboxes for images")
# iterate through all images, apply masks, get boxes, and add them to the df
imgDataPaths = [imagePath, negativePath]

for imgDataPath in imgDataPaths:
    if 'negative' in imgDataPath:
        negative = True
        negSuffix = '_neg'
    else:
        negative = False
        negSuffix = ''
    imgPathlist = Path(imgDataPath).glob('flir_*.jpg')
    for imgPath in imgPathlist:
        filename = os.path.splitext(os.path.basename(str(imgPath)))[0]
        # print(filename)
        if filename:
            img = cv.imread(str(imgPath))
            newImage = resize(img)
            newFileName = regularPath + '/' + filename + negSuffix + '_reg.jpg'
            cv.imwrite(newFileName, newImage)
            rowOtsu, rowBin = BoxImages(otsuMaskedPath, binMaskedPath, otsuBoxPath, binBoxPath, otsuGrayPath, binGrayPath, newImage, filename, negative)
            if rowOtsu:
                rowOtsu["imagePath"] = newFileName
                dataOtsu = dataOtsu.append(rowOtsu, ignore_index=True)
                # print(dataOtsu)
            if rowBin:
                rowBin["imagePath"] = newFileName
                dataBin = dataBin.append(rowBin, ignore_index=True)
                # print(dataBin)

# convert dataframes to json
dataOtsu = dataOtsu.fillna(0)
print("otsu mask dataframe without augmentations")
print(dataOtsu)
dataOtsu.to_json(dataOtsuPath)
dataBin = dataBin.fillna(0)
print("binary mask dataframe without augmentations")
print(dataBin)
dataBin.to_json(dataBinPath)

# Split into train and test
trainOtsuPath, testOtsuPath = dataSplit(dataOtsuPath)
trainBinPath, testBinPath = dataSplit(dataBinPath)

print("Generating Augmentations") # for each image
for imgDataPath in imgDataPaths:
    imgPathlist = Path(imgDataPath).glob('flir_*.jpg')
    for imgPath in imgPathlist:
        filename = os.path.splitext(os.path.basename(str(imgPath)))[0]
        if filename:
            img_color_rgb_resized = resize(image=cv.cvtColor(img, cv.COLOR_BGR2RGB))
            augment(augPath, filename + '_aug' + negSuffix, augtype=['Rotate', 'Flip', 'Shift'],
                    X_train=img_color_rgb_resized, Y_train=None)
            # Not sure what standardize and whiten do to color images. So keeping it separate from
            # other augmentations
            # augment(augPath, filename + '_aug' + negSuffix, augtype=['Standardize','Whiten'], X_train=img_color_rgb_resized, Y_train=None)


dfOtsuTrain = pd.read_json(trainOtsuPath)
dfOtsuTest = pd.read_json(testOtsuPath)
dfBinTrain = pd.read_json(trainBinPath)
dfBinTest = pd.read_json(testBinPath)

print("Generating bboxes for Augmented Images")
augImgPathList = Path(augPath).glob('flir_*.jpg')
for augImgPath in augImgPathList:
    augFileName = os.path.splitext(os.path.basename(str(augImgPath)))[0]
    # augFileName = os.path.basename(str(augImgPath))
    if augFileName:
        img = cv.imread(str(augImgPath))
        if '_neg' in augFileName:
            rowOtsu, rowBin = BoxImages(otsuMaskedPath, binMaskedPath, augBoxOtsuPath, augBoxBinPath,
                                        otsuGrayPath, binGrayPath, img, augFileName, True)
        else:
            rowOtsu, rowBin = BoxImages(otsuMaskedPath, binMaskedPath, augBoxOtsuPath, augBoxBinPath,
                                        otsuGrayPath, binGrayPath, img, augFileName, False)

        rowOtsu["imagePath"] = rowBin["imagePath"] = str(augImgPath)

        # looking for original image in a test set:
        inTestSet = False
        for image in dfOtsuTest.imagePath:
            base = os.path.basename(str(image))
            imageFileName = os.path.splitext(base)[0]
            if imageFileName.split('_')[1] in augFileName:
                inTestSet = True
                break

        if inTestSet:
            dfOtsuTest = dfOtsuTest.append(rowOtsu, ignore_index=True)
            dfBinTest = dfBinTest.append(rowBin, ignore_index=True)
        else:
            dfOtsuTrain = dfOtsuTrain.append(rowOtsu, ignore_index=True)
            dfBinTrain = dfBinTrain.append(rowBin, ignore_index=True)


print("Updating json files with data of Augmented Images")
dfOtsuTrain.to_json(trainOtsuPath, orient='records')
dfOtsuTest.to_json(testOtsuPath, orient='records')
dfBinTrain.to_json(trainBinPath, orient='records')
dfBinTest.to_json(testBinPath, orient='records')



dataInfOtsu = pd.DataFrame(columns=columns)
dataInfBin = pd.DataFrame(columns=columns)

print("Generating bboxes for Inference Images")
infPathlist = Path(inferencePath).glob('flir_*.jpg')
for infPath in infPathlist:
    filename = os.path.splitext(os.path.basename(str(infPath)))[0]
    if filename:
        img = cv.imread(str(infPath))
        newImage = resize(img)

        if '_negative' in filename:
            rowOtsu, rowBin = BoxImages(otsuMaskedPath, binMaskedPath, augBoxOtsuPath, augBoxBinPath, otsuGrayPath, binGrayPath, img, filename, True)
        else:
            rowOtsu, rowBin = BoxImages(otsuMaskedPath, binMaskedPath, augBoxOtsuPath, augBoxBinPath, otsuGrayPath, binGrayPath, img, filename, False)

        if rowOtsu:
            rowOtsu["imagePath"] = filename
            dataInfOtsu = dataInfOtsu.append(rowOtsu, ignore_index=True)
            # print(dataOtsu)
        if rowBin:
            rowBin["imagePath"] = filename
            dataInfBin = dataInfBin.append(rowBin, ignore_index=True)

print("Saving Inference bboxes in json files")

dataInfOtsu = dataOtsu.fillna(0)
dataInfBin = dataOtsu.fillna(0)
print(dataInfOtsu)
print(dataInfBin)
dataInfOtsu.to_json(dataOtsuInferencePath, orient='records')
dataInfBin.to_json(dataBinInferencePath, orient='records')

cv.destroyAllWindows()
