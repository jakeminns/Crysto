import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as fi
import math
import keras
import time
import sys
import decimal
sys.path.append(r'/home/jake/Documents/Programming/Github/Python/')

#from SliceOPy import NetSlice, DataSlice

import keras.backend as K
import tensorflow as tf
from CrystoGen import *
from PatternGen import *

def buildModelConv(input_shape,num_classes):
    
    model = keras.Sequential()
    model.add(keras.layers.Conv3D(12, (3,3,3),input_shape=input_shape,padding="same",data_format= keras.backend.image_data_format()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    #model.add(keras.layers.BatchNormalization())
    #model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPooling3D(pool_size=(2, 2,2),data_format= keras.backend.image_data_format()))
    #    
    model.add(keras.layers.Conv3D(12, (3,3, 3),padding="same",data_format= keras.backend.image_data_format()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    #model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPooling3D(pool_size=(2, 2,2),data_format= keras.backend.image_data_format()))
   # model.add(keras.layers.Dropout(0.25))
    ##    
    #model.add(keras.layers.Conv3D(64, (3,3,3),data_format= K.image_data_format()))
    #model.add(keras.layers.BatchNormalization())
    #model.add(keras.layers.Activation('relu'))
    #777model.add(keras.layers.Conv2D(64,(3,3),data_format=K.image_data_format()))
    #model.add(keras.layers.Activation('relu'))
    #model.add(keras.layers.MaxPooling3D(pool_size=(2,2,2),data_format=keras.backend.image_data_format()))
    model.add(keras.layers.Dropout(0.35))

    #model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.AveragePooling3D(pool_size=(2,2 ,2),data_format= K.image_data_format()))
    #model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
    
    #model.add(keras.layers.Dense(1048))
    #model.add(keras.layers.BatchNormalization())
    #model.add(keras.layers.Activation('relu'))
    #model.add(keras.layers.Dense(456))
    #model.add(keras.layers.Activation('relu'))
    #model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(36))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('sigmoid'))

    return model

def buildModelDense(input_shape,num_classes):
    
    model = keras.Sequential()
    model.add(keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(keras.layers.Dense(35000,input_dim=13500,activation='sigmoid'))
    model.add(keras.layers.Dense(12000,activation='sigmoid'))
    model.add(keras.layers.Dense(1000,activation='sigmoid'))
    model.add(keras.layers.DensapplyMultiPospee(59,activation='sigmoid'))

    return model


def writeHKL(grid,ff,d,theta,LP):

    out = open("out.hkl","w")
    out.write("#  h    k    l     |F|        d       2theta       LP   \n")
    for i in range(0,grid.shape[0]):
        out.write("  "+str('{:3d}'.format(int(grid[i][0])))+"  "+str('{:3d}'.format(int(grid[i][1])))+"  "+str('{:3d}'.format(int(grid[i][2])))+"  "+str('{:06f}'.format(ff[i][0].real))+"  "+str('{:06f}'.format(d[i][0]))+"  "+str('{:06f}'.format(theta[i][0]))+"  "+str('{:06f}'.format(LP[i][0]))+"\n")

cell = CrystoGen()
pattern = PatternGen(cell)
pattern = pattern.calculate_powder_pattern(180.0,0.01,0.1,-0.0001,0.0001,0.001,0.5,0.001,8.0,1.0)


"""
#writeHKL(dd,[0,15],[-15,15],[-15,15])
#
#writeGRD(ff.shape,ff,"reciprocal_cubic",a,b,c,alpha,beta,gamma)


genrateTrainingData(10000,[[0,15],[-15,15],[-15,15],sgInfo,asfInfo])


feat = np.load("feat.npy")
label = np.load("label.npy")
#print(feat.shape,label.shape)

model = buildModelConv((15,30,30, 1),1)
data = DataSlice(Features=feat,Labels=label,Channel_Features=[15,30,30,1],Shuffle = False,Split_Ratio=0.85)
data.channelOrderingFormatFeatures(15,30,30,1)
data.oneHot(36)
model = NetSlice(model,'keras', data)
#model.loadModel('3d_Cubic_conv_simple',customObject=None)
#print(model.summary())
model.compileModel(tf.train.AdamOptimizer(), 'categorical_crossentropy', ['accuracy'])
#print(model.summary())
model.trainModel(Epochs=20,Batch_size=1000,Verbose=1)
#model.generativeDataTrain(buildDensity, BatchSize=200, Epochs=10,Channel_Ordering=(36,36,1,1),Info=sgInfo)
#model.generativeDataTrain(buildDensity3D, BatchSize=3000, Epochs=10,Channel_Ordering_Feat=(30,30,30),funcParams=[60,60,60,sgInfo,4,1],OneHot=36)
model.saveModel("3d_Cubic_conv_simple")
"""