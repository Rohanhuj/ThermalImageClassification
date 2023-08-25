import cv2 as cv
import os
from pathlib import Path
import sys
import pandas as pd
import numpy
from matplotlib import pyplot as plt

'''
imagePath = 'images/flir_20200809T140601.jpg'
img = cv.imread(imagePath)

img_color = img
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# global thresholding
ret1,th1 = cv.threshold(img,140,255,cv.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(img,(5,5),0)
ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=140)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()
#plt.savefig('<savefile>.jpg')
'''

imagePath = 'images'
negativePath = 'images/negative'
binThreshPath = 'images/binary-thresh-images'

Path(binThreshPath).mkdir(parents=True, exist_ok=True)

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
        print(filename)
        if filename:
            img = cv.imread(str(imgPath))
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            #newImage = resize(img)
            saveDirName = binThreshPath + '/' + filename + negSuffix + "/"
            Path(saveDirName).mkdir(parents=True, exist_ok=True)
            for i in range(110,150,5):
                # global thresholding
                ret1, th1 = cv.threshold(img, i, 255, cv.THRESH_BINARY)
                saveFileName = filename + "-" + str(i) + '_thresh.png'
                #cv.imwrite(newFileName, newImage)
                # plot all the images and their histograms
                images = [img, 0, th1]
                titles = ['Original Image', 'Histogram', 'Global Thresholding (v=%d)' %i]
                for i in range(0,1):
                    plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
                    plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
                    plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
                    plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
                    plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
                    plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
                    #plt.show()
                    plt.savefig(saveDirName + saveFileName, format='png')