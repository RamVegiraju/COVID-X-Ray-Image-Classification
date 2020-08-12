import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Activation,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")


covid1 = cv2.imread("/Users/ramvegiraju/Desktop/personalProjects/COVIDX-RayDetection/dataset/train/covid/ryct.2020200034.fig5-day7.jpeg")
normal1 = cv2.imread("/Users/ramvegiraju/Desktop/personalProjects/COVIDX-RayDetection/dataset/train/normal/IM-0033-0001-0001.jpeg")
#print(covid1.shape)
#print(normal1.shape)

#Perform transformations on image prior to model building
image_gen = ImageDataGenerator(rotation_range=30, width_shift_range=.1,
                               height_shift_range=.1, rescale=1/255,
                               shear_range=.2, zoom_range=.2,
                               horizontal_flip=True, fill_mode='nearest')

#print(image_gen.random_transform(covid1))
#plt.imshow(image_gen.random_transform(covid1))

#image_gen.flow_from_directory('/Users/ramvegiraju/Desktop/personalProjects/COVIDX-RayDetection/dataset/train')
def imageClassifierModel():
    model = Sequential()
    model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(160,160,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=(160,160,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=(160,160,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    #Reduce overfitting
    model.add(Dropout(.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    #Compile Model
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    return model

model = imageClassifierModel()
#model.summary()

#Hyperparams
batch_size = 16
input_shape = (160,160,3)

#Training Data
train_image_gen = image_gen.flow_from_directory('/Users/ramvegiraju/Desktop/personalProjects/COVIDX-RayDetection/dataset/train',
                                                target_size=input_shape[:2], 
                                                batch_size= batch_size,
                                                class_mode='binary')

#Test Data
test_image_gen = image_gen.flow_from_directory('/Users/ramvegiraju/Desktop/personalProjects/COVIDX-RayDetection/dataset/test',
                                                target_size=input_shape[:2], 
                                                batch_size= batch_size,
                                                class_mode='binary')


#print(train_image_gen.class_indices) Covid 0 Normal 1

#Fitting Model
results = model.fit_generator(train_image_gen,epochs=20,validation_data=test_image_gen,validation_steps=12)
plt.plot(results.history['accuracy'])
#model.save('/Users/ramvegiraju/Desktop/personalProjects/COVIDX-RayDetection')
#tf.keras.models.save_model(model,'/Users/ramvegiraju/Desktop/personalProjects/COVIDX-RayDetection')


#Predictions
covid_image = image.load_img("/Users/ramvegiraju/Desktop/personalProjects/COVIDX-RayDetection/dataset/test/covid/ryct.2020200028.fig1a.jpeg",
                target_size=(160,160))
covid_image = image.img_to_array(covid_image)
covid_image = np.expand_dims(covid_image,axis=0)
print(model.predict_classes(covid_image)) #returns 0 for COVI

#Function for testing on X-ray images
def predictClass(filePath, targetSize):
    testImage = image.load_img(filePath)
    testImage = np.expand_dims(testImage)
    if testImage == '0':
        print("This X-ray has sign of COVID-19")
    else:
        print("This X-ray does not show signs of COVID-19")
