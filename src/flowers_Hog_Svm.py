import os
import numpy as np
import cv2
from sklearn.svm import SVC # SVM klasifikator
from sklearn.metrics import accuracy_score
import csv

# transformisemo u oblik pogodan za scikit-learn
def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))

shape = 520

'''
    Reading training labels and based on associated picture names reading images from training folder.
    Collecting images and labels in two different lists, but in a corresponding order.
'''
train_flower_images = []
train_flower_labels = []
relative_path = '../train_images/'

with open('../resources/train_labels.csv') as train_labels:
    csv_reader = csv.reader(train_labels, delimiter = ',')
    first_line = True
    for row in csv_reader:
        if first_line:
            print(f'Column names are {", ".join(row)}')
            first_line = False
        else:
            img_path = os.path.join(relative_path, row[0])
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
relative_path = '../test_images/'

with open('../resources/test_labels.csv') as test_labels:
    csv_reader = csv.reader(test_labels, delimiter = ',')
    first_line = True
    for row in csv_reader:
        if first_line:
            print(f'Column names are {", ".join(row)}')
            first_line = False
        else:
            img_path = os.path.join(relative_path, row[0])
            test_flower_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

            resized_image = cv2.resize(test_flower_img, (shape, shape), interpolation=cv2.INTER_NEAREST)
            test_flower_images.append(resized_image)
            test_flower_labels.append(row[1])


shape = 520

print("Images #: ", len(train_flower_images))

'''
    Setting up a HOG descritor with 10 bins, 16 pixels by cell and 5 cells by block
    Images have dimensions of 520x520.
'''
nbins = 25
cell_size = (16, 16)
block_size = (3, 3)

win_size = (shape // cell_size[1] * cell_size[1], shape // cell_size[0] * cell_size[0])
block_size = (block_size[1] * cell_size[1], block_size[0] * cell_size[0])
block_stride = (cell_size[1], cell_size[0])
cell_size = (cell_size[1], cell_size[0])

hog = cv2.HOGDescriptor(_winSize = win_size,
                        _blockSize = block_size,
                        _blockStride = block_stride,
                        _cellSize = cell_size,
                        _nbins = nbins)

'''
    Extracting features from training group using hog.compute
'''
train_features = []
for train_flower_img in train_flower_images:
    resized_image = cv2.resize(train_flower_img, (shape, shape), interpolation=cv2.INTER_NEAREST)
    train_features.append(hog.compute(resized_image))

x_train = reshape_data(np.array(train_features))
y_train = np.array(train_flower_labels)

'''
    Fitting extracted training features inside SVM
'''
clf_svm = SVC(kernel='linear',decision_function_shape = 'ovo')
clf_svm.fit(x_train, y_train)

'''
    Extracting features from test group using hog.compute
'''
test_features = []
for test_flower_img in test_flower_images:
    resized_image = cv2.resize(test_flower_img, (shape, shape), interpolation=cv2.INTER_NEAREST)
    test_features.append(hog.compute(resized_image))

x_test = reshape_data(np.array(test_features))
y_test = np.array(test_flower_labels)


'''
    Training and testing section. Here we are taking predictions from SVM.
    It will go through train images and predict their labels and then go through
    test images and predict their labels.
'''
print("started training...")
training_prediction = clf_svm.predict(x_train)
print("ended training.")
print("\nstarted testing...")
test_prediction = clf_svm.predict(x_test)
print("ended testing.")

'''
    After testing and training we got some accuracy in predicting when we compare it to the actual labels
'''
print("training ", round(accuracy_score(y_train, training_prediction)*100, 2), '%')
print("testing ", round(accuracy_score(y_test, test_prediction)*100, 2) , '%')

'''
    CONCLUSION: current best for test_prediction : 60%
    
    shape = 520
    nbins = 25
    cell_size = (16, 16)
    block_size = (3, 3)
'''