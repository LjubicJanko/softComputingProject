import csv
import os

import cv2
import keras
import numpy as np
from keras.callbacks.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, Conv2D
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


num_epochs = 40
batch_size = 32
shape = 520
num_classes = 5

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

train_flower_images = []
train_flower_labels = []
relative_path = '../train_images/'

with open('../resources/train_labels.csv') as train_labels:
    csv_reader = csv.reader(train_labels, delimiter=',')
    first_line = True
    for row in csv_reader:
        if first_line:
            print(f'Column names are {", ".join(row)}')
            first_line = False
        else:
            img_path = os.path.join(relative_path, row[0])
            train_flower_img = load_image(img_path)

            resized_image = cv2.resize(train_flower_img, (shape, shape), interpolation=cv2.INTER_NEAREST)
            train_flower_images.append(resized_image)
            train_flower_labels.append(row[1])

test_flower_images = []
test_flower_labels = []
relative_path = '../test_images/'

with open('../resources/test_labels.csv') as test_labels:
    csv_reader = csv.reader(test_labels, delimiter=',')
    first_line = True
    for row in csv_reader:
        if first_line:
            print(f'Column names are {", ".join(row)}')
            first_line = False
        else:
            img_path = os.path.join(relative_path, row[0])
            test_flower_img = load_image(img_path)

            resized_image = cv2.resize(test_flower_img, (shape, shape), interpolation=cv2.INTER_NEAREST)
            test_flower_images.append(resized_image)
            test_flower_labels.append(row[1])

label_encoder = LabelEncoder()

train_flower_images = np.array(train_flower_images)
train_flower_labels = to_categorical(label_encoder.fit_transform(train_flower_labels), num_classes=num_classes)

test_flower_images = np.array(test_flower_images)
test_flower_labels = to_categorical(label_encoder.fit_transform(test_flower_labels), num_classes=num_classes)

aug = ImageDataGenerator(rotation_range=10,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode="nearest")

checkpointer = ModelCheckpoint(filepath="winner.hdf5", verbose=1, save_best_only=True)
callbacks_list = [checkpointer]

print("train_img", train_flower_images.shape)
print("train_lab", train_flower_labels.shape)
print("test_img", test_flower_images.shape)
print("test_lab", test_flower_labels.shape)

train = model.fit(train_flower_images,
                  train_flower_labels,
                  batch_size=batch_size,
                  epochs=num_epochs, verbose=1,
                  validation_data=(test_flower_images, test_flower_labels),
                  callbacks=callbacks_list)

test_eval = model.evaluate(test_flower_images, test_flower_labels, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

# hist = model.fit_generator(aug.flow(train_flower_images, train_flower_labels, batch_size=batch_size),
#                            validation_data=(test_flower_images, test_flower_labels),
#                            steps_per_epoch=len(train_flower_images) // batch_size,
#                            callbacks=[checkpointer],
#                            epochs=num_epochs, verbose=1)
