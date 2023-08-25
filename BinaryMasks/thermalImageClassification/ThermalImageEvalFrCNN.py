from __future__ import division
import os
from pathlib import Path
import tensorflow as tf
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2 as cv

import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers

sys.setrecursionlimit(40000)

parser = OptionParser()

#parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
                help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename", help=
                "Location to read the metadata related to the training (generated when training).",
                default="config.pickle")
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')

(options, args) = parser.parse_args()

#if not options.test_path:   # if filename is not given
    #parser.error('Error: path to test data must be specified. Pass --path to command line')

config_output_filename = options.config_filename

with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

if C.network == 'resnet50':
    import keras_frcnn.resnet as nn
elif C.network == 'vgg':
    import keras_frcnn.vgg as nn

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

#img_path = options.test_path

def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height,width,_) = img.shape

    if width <= height:
        ratio = img_min_side/width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side/height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
    return img, ratio

def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    #print("img shape = ", img.shape)
    #print("img ratio = ", ratio)
    img = format_img_channels(img, C)
    #print("img shape after channel formatting = ", img.shape)
    return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2 ,real_y2)

def IOU(bbox1, bbox2):
    #Calculate overlap between two bounding boxes [x, y, w, h] as the area of intersection over the area of unity
    x1, y1, w1, h1 = bbox1[0], bbox1[2],abs(bbox1[1] - bbox1[0]),abs(bbox1[3] - bbox1[2])
    x2, y2, w2, h2 = bbox2[0], bbox2[2], abs(bbox2[1] - bbox2[0]), abs(bbox2[3] - bbox2[2])

    w_I = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_I = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_I <= 0 or h_I <= 0:  # no overlap
        return 0.
    I = w_I * h_I

    U = w1 * h1 + w2 * h2 - I

    return I / U

def loadTrainData(trainCsvPath, gtColumns):
    columnsCsv = ['imagePath','xmin','ymin','xmax','ymax','object_type']
    dfTrainCsv = pd.read_csv(trainCsvPath, sep=',', names=columnsCsv)
    dfData = pd.DataFrame(columns = gtColumns)
    for index,row in dfTrainCsv.iterrows():
        rowTmp = {}
        rowsTmp = dfData[dfData['imagePath']==row.imagePath].to_dict('index')
        for item in rowsTmp:
            rowIndex = item
            rowTmp = rowsTmp[rowIndex]
            newRow = rowTmp
        print("rowTmp = ", rowTmp)
        if not rowTmp:
            newRow = {}
            newRow['imagePath'] = row.imagePath
            newRow['numberOfBoxes'] = 0
        newRow['numberOfBoxes'] += 1
        numBoxes = newRow['numberOfBoxes']
        tmpColumnX1 = f"box{numBoxes}-x1"
        tmpColumnY1 = f"box{numBoxes}-y1"
        tmpColumnX2 = f"box{numBoxes}-x2"
        tmpColumnY2 = f"box{numBoxes}-y2"
        newRow[tmpColumnX1] = row.xmin
        newRow[tmpColumnY1] = row.ymin
        newRow[tmpColumnX2] = row.xmax
        newRow[tmpColumnY2] = row.ymax
        print("newRow = ", newRow)
        if not rowTmp:
            dfData = dfData.append(newRow, ignore_index=True)
        else:
            dfData.loc[rowIndex] = list(newRow.values())
    return dfData

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

    # load the model
    modelPath = "model_frcnn.hdf5"

    class_mapping = C.class_mapping

    if 'bg' not in class_mapping:
        class_mapping['bg'] = len(class_mapping)

    class_mapping = {v: k for k, v in class_mapping.items()}
    print(class_mapping)
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
    C.num_rois = int(options.num_rois)

    if C.network == 'resnet50':
        num_features = 1024
    elif C.network == 'vgg':
        num_features = 512

    if K.image_data_format() == 'classes_first':
        input_shape_img = (3, None, None)
        input_shape_features = (num_features, None, None)
    else:
        input_shape_img = (None, None, 3)
        input_shape_features = (None, None, num_features)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(C.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)

    classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    model_classifier = Model([feature_map_input, roi_input], classifier)

    # If modelPath parameter is specified use it to over-ride the configured model path
    if not modelPath:
        modelPath = C.model_path
    print('Loading weights from {}'.format(modelPath))
    model_rpn.load_weights(modelPath, by_name=True)
    model_classifier.load_weights(modelPath, by_name=True)

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    gtDataPath = 'images/otsu-images/dataframe.json'
    gtTrainPath = os.path.dirname(gtDataPath) + "/train_images"
    gtTrainCsvPath = gtTrainPath + "/annotate.txt"
    resultImgPath = os.path.dirname(gtDataPath) + "/result_plots_frcnn"
    Path(resultImgPath).mkdir(parents=True, exist_ok=True)

    # FrCNN model does not use the standard train and test images
    # Load training data frame for class(column) names i.e. genres
    # Run validation on all the ground-truth images including train and test
    img_path = gtTrainPath
    dfData = pd.read_json(gtDataPath)  # read the groundtruth json file
    columns = np.array(dfData.columns)
    #TODO : Add temporary results file to avoid re-running image prediction multiple times
    tmpResultsJsonPath = os.path.dirname(gtDataPath) + "/tmpResultsFrCNN.json"
    if os.path.exists(tmpResultsJsonPath):
        dfResults = pd.read_json(tmpResultsJsonPath)
    else:
        dfResults = pd.DataFrame(columns=columns)

        all_imgs = []
        classes = {}
        bbox_threshold = 0.8
        visualise = True

        for idx, img_name in enumerate(sorted(os.listdir(img_path))):
            if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                continue
            print(img_name)

            row = {}
            st = time.time()
            filepath = os.path.join(img_path,img_name)
            filename = os.path.splitext(img_name)[0]
            fileExt = os.path.splitext(img_name)[1]
            row["imagePath"] = filepath
            row["numberOfBoxes"] = 0

            img = cv.imread(filepath)

            X, ratio = format_img(img, C)

            if K.image_data_format() == 'channels_last':
                X = np.transpose(X, (0, 2, 3, 1))

            # get the feature maps and output from the RPN
            [Y1, Y2, F] = model_rpn.predict(X)

            R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_data_format(), overlap_thresh=0.7)

            # convert from (x1,y1,x2,y2) to (x,y,w,h)
            R[:, 2] -= R[:, 0]
            R[:, 3] -= R[:, 1]

            # apply the spatial pyramid pooling to the proposed regions
            bboxes = {}
            probs = {}

            for jk in range(R.shape[0]//C.num_rois + 1):
                ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
                if ROIs.shape[1] == 0:
                    break

                if jk == R.shape[0]//C.num_rois:
                    #pad R
                    curr_shape = ROIs.shape
                    target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
                    ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                    ROIs_padded[:, :curr_shape[1], :] = ROIs
                    ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                    ROIs = ROIs_padded

                [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

                for ii in range(P_cls.shape[1]):

                    if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                        continue

                    cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                    if cls_name not in bboxes:
                        bboxes[cls_name] = []
                        probs[cls_name] = []

                    (x, y, w, h) = ROIs[0, ii, :]

                    cls_num = np.argmax(P_cls[0, ii, :])
                    try:
                        (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                        tx /= C.classifier_regr_std[0]
                        ty /= C.classifier_regr_std[1]
                        tw /= C.classifier_regr_std[2]
                        th /= C.classifier_regr_std[3]
                        x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                    except:
                        pass
                    bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
                    probs[cls_name].append(np.max(P_cls[0, ii, :]))

            all_dets = []

            for key in bboxes:
                bbox = np.array(bboxes[key])

                new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
                row["numberOfBoxes"] = len(new_boxes)
                for jk in range(new_boxes.shape[0]):
                    (x1, y1, x2, y2) = new_boxes[jk,:]

                    (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

                    cv.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)
                    tmpColumnX1 = f"box{jk+1}-x1"
                    tmpColumnY1 = f"box{jk+1}-y1"
                    tmpColumnX2 = f"box{jk+1}-x2"
                    tmpColumnY2 = f"box{jk+1}-y2"
                    row[tmpColumnX1] = real_x1
                    row[tmpColumnY1] = real_y1
                    row[tmpColumnX2] = real_x2
                    row[tmpColumnY2] = real_y2

                    textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
                    all_dets.append((key,100*new_probs[jk]))

                    (retval,baseLine) = cv.getTextSize(textLabel,cv.FONT_HERSHEY_COMPLEX,1,1)
                    textOrg = (real_x1, real_y1-0)

                    cv.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
                    cv.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
                    cv.putText(img, textLabel, textOrg, cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

            print('Elapsed time = {}'.format(time.time() - st))
            print(all_dets)
            #cv.imshow('img', img)
            #cv.waitKey(0)
            cv.imwrite('{}/{}_plot.{}'.format(resultImgPath,filename,fileExt),img)
            dfResults = dfResults.append(row, ignore_index=True)

        dfResults = dfResults.fillna(0)

    print("Results shape = ", dfResults.shape)
    print("Results columns = ", dfResults.columns)
    print("Results head = ", dfResults.head())
    #TODO : Add temporary results file to avoid re-running image prediction multiple times
    dfResults.to_json(tmpResultsJsonPath)

    trainCsvPath = gtTrainPath + "/annotate.txt"
    dfTrain = loadTrainData(trainCsvPath, columns)
    dfTrain = dfTrain.fillna(0)
    print("Train shape = ", dfTrain.shape)
    print("Train columns = ", dfTrain.columns)
    print("Train head = ", dfTrain.head())

    #Calculate measured accuracy of model using IOU[intersection over union]
    #compute IOU for y_test vs y_pred for each corresponding bounding box
    #print out accuracy for each imagePath
    #finally print out average accuracy across all test images

    # Initialize new result column with zeros
    dfResults['IOU'] = 0

    for index,row in dfTrain.iterrows():
        boxTrain = np.array(dfTrain.iloc[index,2:6])
        boxPred = np.array(dfResults.loc[dfResults['imagePath'] == row.imagePath,:].values.flatten().tolist()[2:6])
        print("Computing IOU for imagePath ", row.imagePath, ", boxTrain ", boxTrain,", vs boxPred ", boxPred)
        iou = IOU(boxTrain,boxPred)
        dfResults.loc[dfResults['imagePath'] == row.imagePath,'IOU'] = iou
        #TODO : Add support for computing IOU for second box also
    print("Average IOU (Metric : Higher is better, max=1) = ", dfResults['IOU'].mean())

    resultsJsonPath = os.path.dirname(gtDataPath) + "/resultsFrCNN.json"
    dfResults.to_json(resultsJsonPath)

    #TODO : in case number of boxes is a mismatch, print as a separate accuracy metric

    #TODO : Add MinRotatedRectangles to predict Rotated images better

