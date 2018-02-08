"""
Utilities for the data science bowl
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


### DATA LOADING ###

# Load all the train masks
def get_train_masks():
	masks = np.array([])
	for file_name in glob.iglob('stage1_train/*/masks/*'):
	    im = imageio.imread(file_name)
	    im = rgb2gray(im)
	    masks = np.append(masks, im, axis=0)
    return masks

# Load all the train images
def get_train_images():
	images = np.array([])
	for file_name in glob.iglob('stage1_train/*/image/*.png'):
		im = imageio.imread(file_name)
		images = np.append(images, im, axis=0)
	return images

# Load all the test images
def get_train_images():
	images = np.array([])
	for file_name in glob.iglob('stage1_test/*/image/*.png'):
		im = imageio.imread(file_name)
		images = np.append(images, im, axis=0)
	return images



# Given a mask image, returns the coordinates of nuclei
# i.e. pixels that are non-0
def get_nuclei_pixels(image):
    return np.argwhere(image != 0)



