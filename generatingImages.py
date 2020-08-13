#Not enough X-ray images using GAN to generate
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
import cv2
import os
from matplotlib.image import imread
import numpy as np

#Checking for average dimensions of images before building model
def checkDimensions(imageDir):
    """Average Dimensions (1100,1270)"""
    dim1 = []
    dim2 = []
    fileList = os.listdir(imageDir)
    for f in fileList:
        imgs = imread(f)
        d1,d2,colors = imgs.shape
        dim1.append(d1)
        dim2.append(d2)
        firstDim = np.mean(dim1)
        secondDim = np.mean(dim2)
    return firstDim, secondDim
#print(checkDimensions(directory)) 


#Discriminator model identifies whether image is fake or not
def createDiscriminator(inp_shape = (160,160,3)):
    discriminator = Sequential()
    discriminator.add(Flatten(input_shape=inp_shape[:2]))
    discriminator.add(Dense(150,activation='relu'))
    discriminator.add(Dense(100,activation='relu'))
    discriminator.add(Dense(1,activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer='adam')
    return discriminator

# define model
discriminatorModel = createDiscriminator()
# summarize the model
#discriminatorModel.summary()

#Generator model
normal1 = cv2.imread("/Users/ramvegiraju/Desktop/personalProjects/COVIDX-RayDetection/dataset/train/normal/IM-0033-0001-0001.jpeg")
#print(normal1.shape)

directory = "/Users/ramvegiraju/Desktop/personalProjects/COVIDX-RayDetection/dataset/train/covid"
#print(os.listdir(directory))
#for filename in directory:
    #print(filename)
    #imgs = cv2.imread(filename)
    #dims = imgs.shape
    #dimsList.append(dims)

