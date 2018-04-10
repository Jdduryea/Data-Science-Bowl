# source: https://www.kaggle.com/sjoly123/kerasunet-281

# Neural net model

# Native Python stuff
import os
import sys
import random
import warnings

# Matrix ops
import numpy as np
import pandas as pd
import scipy

# Plotting
import matplotlib.pyplot as plt

# Image analysis
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images

from skimage.transform import resize
from skimage.morphology import label
from scipy.ndimage.filters import gaussian_laplace

# Neural Network auto-differentiators
from keras.layers import *
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
import tensorflow as tf
import keras.losses

# TODO:
# Maybe use LeakyRelu?

# Source:
# This code is based on the work done by Peter Grenholm
# https://www.kaggle.com/toregil/a-lung-u-net-in-keras

# Scoring metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)





# source: https://www.kaggle.com/c/ultrasound-nerve-segmentation/discussion/21358
def dice_coef(y_true, y_pred):
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    smooth=0.1
    intersection = 2. * K.sum(y_true_f * y_pred_f, axis=1, keepdims=True) + smooth
    union = K.sum(y_true_f, axis=1, keepdims=True) + K.sum(y_pred_f, axis=1, keepdims=True) + smooth
    return K.mean(intersection / union)


# def gen_data(X_train, y_train, batch_size):
# 	data_gen = ImageDataGenerator(
# 			rotation_range = 90,
# 			shear_range=0.2,
# 			zoom_range=0.2,
# 			horizontal_flip=True,
# 			vertical_flip=True,
# 			fill_mode='nearest').flow(X_train,X_train, batch_size,seed=1)

# 	mask_gen = ImageDataGenerator(
# 		rotation_range = 90,
# 		shear_range=0.2,
# 		zoom_range=0.2,
# 		horizontal_flip=True,
# 		vertical_flip=True,
# 		fill_mode='nearest').flow(y_train,y_train, batch_size, seed=1)

# 	while 1:
# 		X_data, nil = data_gen.next()
# 		y_data, nil = mask_gen.next()
# 		yield X_data, y_data

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

# Unet class, can be trained on data and then predicted
class UNET():
	def __init__(self, X_train, I_WIDTH=128, I_HEIGHT=128, I_CHN=1):
		self.model = None
		self.history = None
		self.IMG_WIDTH = I_WIDTH
		self.IMG_HEIGHT = 128
		self.IMG_CHANNELS = I_CHN

		# Set up network architecture
		input_layer = Input(shape=X_train.shape[1:])
		c1 = Conv2D(filters=8, kernel_size=(3,3), activation="relu", padding='same')(input_layer)
		l = MaxPooling2D(strides=(2,2))(c1)
		c2 = Conv2D(filters=16, kernel_size=(3,3), activation="relu", padding='same')(l)
		l = MaxPooling2D(strides=(2,2))(c2)
		c3 = Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding='same')(l)
		l = MaxPooling2D(strides=(2,2))(c3)
		c4 = Conv2D(filters=32, kernel_size=(1,1), activation="relu", padding='same')(l)
		l = concatenate([UpSampling2D(size=(2,2))(c4), c3], axis=-1)
		l = Conv2D(filters=32, kernel_size=(2,2), activation="relu", padding='same')(l)
		l = concatenate([UpSampling2D(size=(2,2))(l), c2], axis=-1)
		l = Conv2D(filters=24, kernel_size=(2,2), padding='same')(l)
		l = concatenate([UpSampling2D(size=(2,2))(l), c1], axis=-1)
		l = Conv2D(filters=16, kernel_size=(2,2), activation="relu", padding='same')(l)
		l = Conv2D(filters=64, kernel_size=(1,1), activation="relu")(l)
		l = Dropout(0.5)(l)
		output_layer = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid')(l)
		                                                         
		self.model = Model(input_layer, output_layer)
		self.model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=[mean_iou])


	def train(self, X_train, y_train, X_val, y_val):

		# data augmentation
		batch_size = 8

		# datagen.fit(X_train)
		# mask_datagen.fit(y_train)

		num_epochs = 10

		#earlystopper = EarlyStopping(patience=5, verbose=1)
		#checkpointer = ModelCheckpoint('unet_weights_1.h5', monitor='val_loss', verbose=1, save_best_only=True)
		print "done setting up"
		# fits the model on batches with real-time data augmentation:
		# self.history = self.model.fit_generator(my_gen, 
		# 	validation_data = (X_val,y_val), epochs=num_epochs,callbacks=[earlystopper, checkpointer], steps_per_epoch=30)


		# earlystopper = EarlyStopping(patience=5, verbose=1)
		# checkpointer = ModelCheckpoint('unet_weights_1.h5', monitor='val_loss', verbose=1, save_best_only=True)
		#self.history = self.model.fit(X_train, y_train, validation_split=0.1, batch_size=batch_size, epochs=num_epochs, callbacks=[earlystopper, checkpointer])
		self.history = self.model.fit(X_train, y_train, validation_split=0.1, batch_size=8, epochs=num_epochs)


	def load_weights(self,weight_file):

		keras.losses.custom_loss = mean_iou
		model = load_model(weight_file, custom_objects={'mean_iou': mean_iou})


	def predict(self, X):
		return self.model.predict(X)









