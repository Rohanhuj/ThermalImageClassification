import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import cv2 as cv
from sklearn.model_selection import train_test_split


print(tf.__version__)
print(keras.__version__)



data = pd.read_json (r'jsonData/dataframe.json')
# print(images_data.columns)

labels = data.iloc[:, 2:] # coordinates of corners
# y = np.ravel(labels) # converts into an array (but then it loses structure?...)

# print(labels)

X = [] # contains all the image data
for img_path in data.imagePath:
	img = cv.imread(str(img_path))
	resized_image = cv.resize(img, (540, 720))

	imageArray = np.asarray(resized_image)
	# print(img_path)
	# print(imageArray.shape)
	X.append(imageArray)

X = np.asarray(X)
X_train_full, X_test, y_train_full, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

print(X_train_full.shape)
print()
print(X_test.shape)
print()

X_valid, X_train = X_train_full[:300], X_train_full[300:]
y_valid, y_train = y_train_full[:300], y_train_full[300:]

print(X_valid.shape)


keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

number_of_bbox_values = labels.shape[1] # bbox means bounding box

# copied from classification model (needs to be changed)
model = keras.models.Sequential([
    # keras.layers.Flatten(input_shape=[540, 720]),
    keras.layers.Dense(30, activation="relu", input_shape=(540, 720)),
    # keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(number_of_bbox_values, activation="softmax")
])

# model.compile(loss="sparse_categorical_crossentropy",
#               optimizer="sgd",
#               metrics=["accuracy"])

model.summary()


# model.compile(loss="sparse_categorical_crossentropy",
#               optimizer="sgd",
#               metrics=["accuracy"])
#
# history = model.fit(X_train, y_train, epochs=2,
#                     validation_data=(X_valid, y_valid))
#
# model.evaluate(X_test, y_test)
#
# X_new = X_test[:3]
# y_proba = model.predict(X_new)
# y_proba.round(2)


"""
labels = images_data['imagePath']
y = np.ravel(labels) # converts into a np array

X = images_data.iloc[:,1:] # number data that is useful

print(y)
print()
print(X)
print()

# separate data into training and testing data (X and y)
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_valid, X_train = X_train_full[:300] / 1., X_train_full[300:] / 1.
y_valid, y_train = y_train_full[:300], y_train_full[300:]
X_test = X_test / 1.

print(X_valid.shape)

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

# model = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[540, 720]),
#     # keras.layers.Dense(300, activation="relu"),
#     keras.layers.Dense(100, activation="relu"),
#     keras.layers.Dense(10, activation="softmax")
# ])
"""





sys.exit()


fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.


keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=2,
                    validation_data=(X_valid, y_valid))

model.evaluate(X_test, y_test)

X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)

y_pred = model.predict_classes(X_new)
np.array(class_names)[y_pred]

plt.figure(figsize=(7.2, 2.4))
for index, image in enumerate(X_new):
    plt.subplot(1, 3, index + 1)
    plt.imshow(image, cmap="binary", interpolation="nearest")
    plt.axis('off')
    plt.title(class_names[y_test[index]], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
# save_fig('fashion_mnist_images_plot', tight_layout=False)
plt.show()


# copied from moviePosterTrain
# model = Sequential()
# model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(400,400,3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(25, activation='sigmoid'))