"""
Models for the data science bowl
Jack Duryea jd50
Shashank Mahesh sm103
"""

import numpy as np
import os
import sys
import pathlib
import imageio
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from scipy import stats
import glob
import data_utils as du
import time
from sklearn.metrics import confusion_matrix
from skimage import filters
from skimage import exposure
from skimage.morphology import watershed
from skimage.filters import sobel
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.segmentation import random_walker
from skimage.data import binary_blobs
from skimage.exposure import rescale_intensity
import skimage


### MODELS ###

# Converts an image to black and white,
# then uses a threshold to label nuclei.
# Image is a du.FullImage
# Returns the mask
def naive_threshold(image, thresh = 0.1):
	bw = rgb2gray(image.im)
	bw2 = np.where(bw > thresh, 1, 0)
	return bw2

# Uses Otsu thresholding to label nuclei.
# Returns the mask.
def otsu_threshold(image):
	bw = rgb2gray(image.im)
	val = filters.threshold_otsu(bw)
	return bw > val

def random_walk_threshold(image):
	bw = rgb2gray(image.im)
	markers = np.zeros(bw.shape, dtype=np.uint)
	markers[bw < 0.5] = 0
	markers[bw > 0.5] = 1

	# Run random walker algorithm
	labels = random_walker(bw, markers, beta=10, mode='bf')
	return labels


### Scoring Metrics ###

# Finds the confusion matrix of the prediction
def score(true, predicted):
	t = true.flatten()
	t[t > 1] = 1

	p = predicted.flatten()
	p[p > 1] = 1
	tn, fp, fn, tp = confusion_matrix(t, p,labels=[1,0]).ravel()
	acc = (tn + tp)/float(tn + fp + fn + tp)
	return acc, tn, fp, fn, tp, confusion_matrix(t, p,labels=[1,0])

# returns accuracy, precision, recall, and f1 score
def get_metrics(tps, fps, tns, fns):
	a = accuracy(tps, fps, tns, fns)
	p = precision(tps, fps, tns, fns)
	r = recall(tps, fps, tns, fns)
	f1_score = f1(tps, fps, tns, fns) 
	print "Accuracy:  ", a
	print "Precision: ", p
	print "Recall:    ", r
	print "F1 Score:  ", f1_score
	return a, p, r, f1_score

# Gets tne accuracy of a model based on its confusion matrix
def accuracy(tps, fps, tns, fns):
	return (tps+tns)/float(tps+tns + fps + fns)

# Computes the precision score
def precision(tps, fps, tns, fns):
	return tps/float(tps + fps)

# Computes the recall score
def recall(tps, fps, tns, fns):
	return tps/float(tps + fns)

# Computes the F1 score
def f1(tps, fps, tns, fns):
	p = precision(tps, fps, tns, fns)
	r = recall(tps, fps, tns, fns)
	return 2.0 * ((p*r)/float(p+r))


