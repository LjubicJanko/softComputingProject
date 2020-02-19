import csv
import os

import cv2
import keras
import numpy as np
from keras.layers import Dense, Flatten, Conv2D
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

'''
    Images will be resized to 100x100 resolution. There are 5 different classes of flowers and the classification
    will be processed in 40 epochs with batch size of 32
'''
num_epochs = 20
batch_size = 16
shape = 100
num_classes = 5

'''
    Network model consists of multiple layers.
    
    It is compiled with adam optimizer.
'''
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(shape, shape, 3), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2), padding='same'))

model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

'''
    Reading training labels and based on associated picture names reading images from training folder.
    Collecting images and labels in two different lists, but in a corresponding order.
'''
train_flower_images = []
train_flower_labels = []
train_flower_path = '../train_images/'

with open('../resources/train_labels.csv') as train_labels:
    csv_reader = csv.reader(train_labels, delimiter=',')
    first_line = True
    for row in csv_reader:
        if first_line:
            first_line = False
        else:
            img_path = os.path.join(train_flower_path, row[0])
            train_flower_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

            resized_image = cv2.resize(train_flower_img, (shape, shape), interpolation=cv2.INTER_NEAREST)
            train_flower_images.append(resized_image)
            train_flower_labels.append(row[1])

'''
    Reading test labels and based on associated picture names reading images from training folder.
    Collecting images and labels in two different lists, but in a corresponding order.
'''
test_flower_images = []
test_flower_labels = []
test_flower_path = '../test_images/'

with open('../resources/test_labels.csv') as test_labels:
    csv_reader = csv.reader(test_labels, delimiter=',')
    first_line = True
    for row in csv_reader:
        if first_line:
            first_line = False
        else:
            img_path = os.path.join(test_flower_path, row[0])
            test_flower_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

            resized_image = cv2.resize(test_flower_img, (shape, shape), interpolation=cv2.INTER_NEAREST)
            test_flower_images.append(resized_image)
            test_flower_labels.append(row[1])

label_encoder = LabelEncoder()

train_flower_images = np.array(train_flower_images)  # x for training
train_flower_labels = to_categorical(label_encoder.fit_transform(train_flower_labels),
                                     num_classes=num_classes)  # y for training

test_flower_images = np.array(test_flower_images)  # x for testing
test_flower_labels = to_categorical(label_encoder.fit_transform(test_flower_labels),
                                    num_classes=num_classes)  # y for testing


train_flower_images = train_flower_images / 255
test_flower_images = test_flower_images / 255


'''
    Fitting training data and labels inside model.
'''
train = model.fit(train_flower_images,
                  train_flower_labels,
                  batch_size=batch_size,
                  epochs=num_epochs,
                  verbose=1,
                  validation_split=0.3)

'''
    Evaluating test data and labels 
'''
test_eval = model.evaluate(test_flower_images, test_flower_labels, verbose=1)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1]*100 , "%")

'''
    Best prediction achieved: 86.25%
'''