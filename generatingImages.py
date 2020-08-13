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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
from matplotlib.image import imread
import numpy as np

#Directory with COVID-infected images
directory = "/Users/ramvegiraju/Desktop/personalProjects/COVIDX-RayDetection/dataset/train/covid"

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

#Preparing training data for GAN
def prepareTrainingData(imageDir):
    data = []
    fileList = os.listdir(imageDir)
    for ims in fileList:
        image = cv2.imread(ims)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        data.append(image)
    return data
trainingData = prepareTrainingData(directory)
trainingData = np.array(trainingData)/255.0

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


#Generator Model
#Average dimensions of (1100, 1270) for images
coding_size = 200

def createGenerator():
    generator = Sequential()
    generator.add(Dense(200,activation='relu',input_shape=[coding_size]))
    generator.add(Dense(300,activation='relu'))
    generator.add(Dense(1100,activation='relu'))
    generator.add(Reshape([160,160]))
    return generator #not compiling

#define Generator model
generatorModel = createGenerator()

#Setup for training
GAN = Sequential([generatorModel,discriminatorModel])
discriminatorModel.trainable = False
GAN.compile(loss='binary_crossentropy',optimizer='adam')
#GAN.layers[0]
#GAN.layers[1]

#Hyperparams
batch_size = 20
epochs = 1

#Prepare Data
my_data = trainingData
dataset = tf.data.Dataset.from_tensor_slices(my_data).shuffle(buffer_size=1000)
#print(type(dataset))
dataset = dataset.batch(batch_size,drop_remainder=True).prefetch(1)

#layers of GAN
generator,discriminator = GAN.layers
for epoch in range(epochs):
    print(f"Currently on Epoch {epoch+1}")
    i = 0
    for X_batch in dataset:
        i = i + 1
        if i%100 == 0:
            print(f"\t Currently on batch number {i} of {len(my_data)//batch_size}")

        #Discriminator training
        noise = tf.random.normal(shape=[batch_size,coding_size]) #produce fake images
        gen_images = generator(noise)
        X_fake_vs_real = tf.concat([gen_images,tf.dtypes.cast(X_batch,tf.float32)],axis=0)
        y1 = tf.constant([[0.0]]*batch_size + [[1.0]]*batch_size)
        discriminator.trainable = True
        discriminator.train_on_batch(X_fake_vs_real,y1)

        #Generator
        noise = tf.random.normal(shape=[batch_size,coding_size])
        y2 = tf.constant([[1.0]] * batch_size)
        discriminator.trainable = False
        GAN.train_on_batch(noise,y2)
