#Detecting 3 kinds of Pneumonia - 93.9% accuracy on test set, 93.9% on other metrics (precision, recall, f1score)
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Sequential, save_model, load_model, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

imagePaths = []
for dirname, _, filenames in os.walk('./data/'):
    for filename in filenames:
        if (filename[-3:] == 'png' or filename[-4:] == 'jpeg'):
            imagePaths.append(os.path.join(dirname, filename))

imgSize = 28

X = []
Y = []
hmap = {'VIRALPNEUMONIA': 'Viral Pneumonia', 'BACTPNEUMONIA': 'Bacterial Pneumonia', 'NORMAL': 'Normal', 'COVID-19': 'Covid-19'}
for imagePath in tqdm(imagePaths):
    label = imagePath.split(os.path.sep)[-2]
        
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (imgSize, imgSize))

    X.append(image)
    Y.append(hmap[label])

print('Covid-19:',Y.count('Covid-19'))
print('Normal:',Y.count('Normal'))
print('Viral Pneumonia: ',Y.count('Viral Pneumonia'))
print('Bacterial Pneumonia: ',Y.count('Bacterial Pneumonia'))

# encode class values as integers
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = to_categorical(Y)

(trainX, testX, trainY, testY) = train_test_split(X, Y, test_size=0.20, stratify=Y, random_state=42)
(trainX, valX, trainY, valY) = train_test_split(trainX, trainY, test_size=0.25, random_state=42)
del X
del Y

ntimes = 6
trainY = trainY.tolist()
for i in tqdm(range(len(trainX))):
    if (trainY[i][0] == 1):
        trainX += [trainX[i]]*ntimes
        trainY += [trainY[i]]*ntimes
        
trainY = np.array(trainY)

trainX = np.array(trainX).astype('float16')/255

valX = np.array(valX).astype('float16')/255

testX = np.array(testX).astype('float16')/255

trainAug = ImageDataGenerator(rotation_range=20, 
                            horizontal_flip = True,
                            fill_mode="nearest",
                            vertical_flip=True)

es = EarlyStopping(patience = 1500, 
                        monitor = "val_accuracy", 
                        mode="max", 
                        verbose = 1)

# checkpoint to save model
chkpt = ModelCheckpoint(filepath="model.hdf5", 
                        save_best_only=True,
                        monitor = "val_accuracy",
                        mode = "max",
                        verbose=1)
"""
baseModel = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(imgSize, imgSize, 3), classes=4)
model = Sequential()
model.add(baseModel)
model.add(Flatten())
model.add(Dense(512, activation = 'relu',
                    kernel_initializer="he_uniform"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(4, activation = 'softmax'))
for layer in baseModel.layers:
    layer.trainable = False
INIT_LR = 3e-4 #0.175 
epochs = 100
opt = Adam(lr=INIT_LR, decay=INIT_LR / epochs) 
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
"""
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(imgSize, imgSize, 3)))
model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(4, activation='softmax'))

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

epochs=3000
BS = 128 #16
"""
history = model.fit(
    trainAug.flow(trainX, trainY, batch_size=BS, shuffle=True),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(valX, valY),
    validation_steps=len(valX) // BS,
    callbacks = [es,chkpt],
    epochs=epochs,
    workers=8,
    class_weight={0:1.0/2538, 1:1.0/219, 2:1.0/2924, 3:1.0/1345})

#Plots accuracy over epochs
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
"""

model= load_model('./model/model.hdf5')

#Against Train Set
predIdxs = model.predict(trainX, batch_size=BS, verbose=1)
predIdxs = np.argmax(predIdxs, axis=1)

print('Acc: {}'.format(accuracy_score(y_true = trainY.argmax(axis=1), y_pred = predIdxs)))
print('Recall: {}'.format(recall_score(y_true = trainY.argmax(axis=1), y_pred = predIdxs, average='weighted')))
print('Precision: {}'.format(precision_score(y_true = trainY.argmax(axis=1), y_pred = predIdxs, average='weighted')))
print('f1: {}'.format(f1_score(y_true = trainY.argmax(axis=1), y_pred = predIdxs, average='weighted')))

print(confusion_matrix(trainY.argmax(axis=1), predIdxs))
print(classification_report(trainY.argmax(axis=1), predIdxs, target_names=le.classes_, digits = 4))

#Against Val Set
predIdxs = model.predict(valX, batch_size=BS, verbose=1)
predIdxs = np.argmax(predIdxs, axis=1)

print('Acc: {}'.format(accuracy_score(y_true = valY.argmax(axis=1), y_pred = predIdxs)))
print('Recall: {}'.format(recall_score(y_true = valY.argmax(axis=1), y_pred = predIdxs, average='weighted')))
print('Precision: {}'.format(precision_score(y_true = valY.argmax(axis=1), y_pred = predIdxs, average='weighted')))
print('f1: {}'.format(f1_score(y_true = valY.argmax(axis=1), y_pred = predIdxs, average='weighted')))

print(confusion_matrix(valY.argmax(axis=1), predIdxs))
print(classification_report(valY.argmax(axis=1), predIdxs, target_names=le.classes_, digits = 4))

#Against Test Set
predIdxs = model.predict(testX, batch_size=BS, verbose=1)
predIdxs = np.argmax(predIdxs, axis=1)

print('Acc: {}'.format(accuracy_score(y_true = testY.argmax(axis=1), y_pred = predIdxs)))
print('Recall: {}'.format(recall_score(y_true = testY.argmax(axis=1), y_pred = predIdxs, average='weighted')))
print('Precision: {}'.format(precision_score(y_true = testY.argmax(axis=1), y_pred = predIdxs, average='weighted')))
print('f1: {}'.format(f1_score(y_true = testY.argmax(axis=1), y_pred = predIdxs, average='weighted')))

print(confusion_matrix(testY.argmax(axis=1), predIdxs))
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=le.classes_, digits = 4))
"""
import tensorflow as tf
from tensorflow import saved_model

# The export path contains the name and the version of the model
tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
model= load_model('model.hdf5')
export_path = '../covid-pneumonia/1'

# Feth the Keras session and save the model
with tf.compat.v1.Session() as sess:
    tf.saved_model.save(
        sess,
        export_path,
        inputs={'images': model.input},
        outputs={t.name:t for t in model.outputs})
"""