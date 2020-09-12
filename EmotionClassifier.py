# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 22:13:26 2020

@author: Divesh
"""
#influenced by https://github.com/code-by-dt/emotion_detection
#Model inspired from LeCun's LeNet-5 https://ieeexplore.ieee.org/abstract/document/726791
###################################################
# instructions to run this file
#
# Execute the program block by block
# the blocks are divided using a two strings of '#'
# with the block header in between
###################################################

###################################################
#library import
###################################################
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Activation,BatchNormalization,Flatten,Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
###################################################
#gpu check
###################################################
tf.test.is_gpu_available()
###################################################
#image directory and augmentation
###################################################
image_height,image_width=128,128
number_of_classes=8
batch_size=32

#data_dir='E:/UoB/Project/Databases/all_expressions/ck+ext'
#use above path when all files are in a single directory
#use below path when training and testing data are seperate
train_data_dir='E:/UoB/Project/Databases/all_expressions/subsetsck+/testonset6'
valid_data_dir='E:/UoB/Project/Databases/all_expressions/subsetsck+/set6'

#https://keras.io/api/preprocessing/image/#imagedatagenerator-class
train_datagen=ImageDataGenerator(rotation_range=30, 
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 width_shift_range=0.3,
                                 height_shift_range=0.3,
                                 fill_mode="nearest",
                                 horizontal_flip=True,
                                 rescale=1./255)
#                               Enable when using the single path directory
#                                 validation_split=0.2)

#https://keras.io/api/preprocessing/image/#flowfromdirectory-method
#Augmenting the images from the training directory
train_generator=train_datagen.flow_from_directory(train_data_dir,
                                                  target_size=(image_height,image_width),
                                                  color_mode="grayscale",
                                                  class_mode="categorical",
                                                  batch_size=batch_size,
                                                  shuffle=True,
#                               Enable when using the single path directory
#                                                  subset="training",
                                                  interpolation="nearest")

valid_datagen = ImageDataGenerator(rescale=1./255)
#loading the images that need to be tested on
valid_generator=valid_datagen.flow_from_directory(valid_data_dir,
                                                  target_size=(image_height,image_width),
                                                  color_mode="grayscale",
                                                  class_mode="categorical",
                                                  batch_size=batch_size,
                                                  shuffle=True,
#                               Enable when using the single path directory
#                                                  subset="validation",
                                                  interpolation="nearest")
###################################################
#hyper-parameter section 1 
###################################################
train_filenames=train_generator.filenames
valid_filenames=valid_generator.filenames
train_samples=len(train_filenames)
valid_samples=len(valid_filenames)
epochs=666
learning_rate=0.001
file_name='EmotionDetector_teston6-1.h5'
###################################################
#the model
###################################################
#https://keras.io/guides/sequential_model/
#initialising sequential model
cnn=Sequential()
#INPUT
#C1 layer
cnn.add(Conv2D(32,(5,5),input_shape=(image_height,image_width,1)))
cnn.add(Activation('elu'))
cnn.add(BatchNormalization())
#S2 layer
cnn.add(MaxPooling2D(pool_size=(2,2)))
#C3 layer
cnn.add(Conv2D(64,(5,5),input_shape=(image_height,image_width,1)))
cnn.add(Activation('elu'))
cnn.add(BatchNormalization())
#S4 layer
cnn.add(MaxPooling2D(pool_size=(2,2)))
#kernel size change
#C5 layer
cnn.add(Conv2D(128,(7,7),input_shape=(image_height,image_width,1)))
cnn.add(Activation('elu'))
cnn.add(BatchNormalization())
#S6 layer
cnn.add(MaxPooling2D(pool_size=(2,2)))
#C7 layer
cnn.add(Conv2D(256,(7,7),input_shape=(image_height,image_width,1)))
cnn.add(Activation('elu'))
cnn.add(BatchNormalization())
#S8 layer
cnn.add(MaxPooling2D(pool_size=(2,2)))
#flattening the data
cnn.add(Flatten())
#full connection
#F9 layer
cnn.add(Dense(image_height*image_width))
cnn.add(Activation('elu'))
cnn.add(BatchNormalization())
#F10 layer
cnn.add(Dense(image_height+image_width))
cnn.add(Activation('elu'))
cnn.add(BatchNormalization())
#softmax to normalise the values between 0 and 1
#OUTPUT layer
cnn.add(Dense(number_of_classes))
cnn.add(Activation('softmax'))
#printing the summary
print(cnn.summary())
###################################################
#initialising callbacks and loss function
###################################################
#https://www.tensorflow.org/api_docs/python/tf/keras/Model
#https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
#https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy
cnn.compile(optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
#https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau
reduce_lr=ReduceLROnPlateau(monitor='val_accuracy',
                            factor=0.2,
                            patience=24,
                            verbose=1,
                            mode='max',
                            min_delta=0.0001)
#https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
cnn_checkpoint=ModelCheckpoint(file_name,
                               monitor='val_accuracy',
                               verbose=1,
                               save_best_only=True,
                               mode='max')
#https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
early_stop=EarlyStopping(monitor='val_accuracy',
                         min_delta=0,
                         patience=66,
                         verbose=1,
                         mode="max",
                         restore_best_weights=True)

callbacks=[reduce_lr,cnn_checkpoint,early_stop]
###################################################
#fitting the model
###################################################
#https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
history=cnn.fit(train_generator,
                epochs=epochs,
                verbose=1,
                callbacks=callbacks,
                validation_data=valid_generator,
                steps_per_epoch=train_samples//batch_size,
                validation_steps=valid_samples//batch_size,
                use_multiprocessing=True)
###################################################
#plot the graphs
###################################################
import matplotlib.pyplot as plt
#accuracy plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train','test'],loc='upper left')
plt.figure()
#loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train','test'],loc='upper left')
plt.figure()
