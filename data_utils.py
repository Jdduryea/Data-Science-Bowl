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

import time


### DATA LOADING ###


# A class for encapsulating data for a full image
class FullImage():
	# im: the actual image, the result of calling imageio.imread()
	# path: the path to im, including .png
	# dir_id: the path to the parent directory, useful to connect masks the their corresponding full image
	def __init__(self, im, path, dir_id):
		self.im = im
		self.path = path
		self.dir_id = dir_id


# A class for encapsulating data for a mask
class Mask():
	# im: the actual mask, the result of calling imageio.imread()
	# path: the path to im, including .png
	# dir_id: the path to the parent directory, useful to connect masks the their corresponding full image
	def __init__(self, im, path, dir_id):
		self.im = im
		self.path = path
		self.dir_id = dir_id


# Load all the train images

def get_train_images():
	#images = np.array([])
	images = []
	for path in glob.iglob('stage1_train/*/images/*'):
		im = imageio.imread(path)
		#im = rgb2gray(im)

		dir_id = path[:path.index("image")]
		full_image = FullImage(im, path, dir_id)
		images.append(full_image)
		#images = np.append(images, im, axis=0)
	return images


# Load all the train masks
def get_train_masks():
	#masks = np.array([])
	masks = []
	for path in glob.iglob('stage1_train/*/masks/*'):
		im = imageio.imread(path)
		#im = rgb2gray(im)
		#masks = np.append(masks, im, axis=0)
		dir_id = path[:path.index("mask")]
		m = Mask(im, path, dir_id)
		masks.append(m)
	return masks

# Load all the test images
def get_test_images():
	#images = np.array([])
	images = []
	for path in glob.iglob('stage1_test/*/images/*.png'):
		im = imageio.imread(path)
		#im = rgb2gray(im)

		dir_id = path[:path.index("image")]
		full_image = FullImage(im, path, dir_id)
		images.append(full_image)
	return images


# Given a mask image, returns the coordinates of nuclei
# i.e. pixels that are non-0
def get_nuclei_pixels(image):
    return np.argwhere(image != 0)

### PIXEL ENCODING ### 
# source:
# Sam StainsbyFast, tested RLE and input routines
# https://www.kaggle.com/stainsby/fast-tested-rle-and-input-routines


def encode(mask):
    pixels = mask.T.flatten()
    # We need to allow for cases where there is a '1' at either end of the sequence.
    # We do this by padding with a zero at each end when needed.
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)








