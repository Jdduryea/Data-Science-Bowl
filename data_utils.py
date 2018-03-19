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
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
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



def get_total_mask(image, train_masks):
	associated_masks = []
	for m in train_masks:
		if m.dir_id == image.dir_id:
			associated_masks.append(m)
	return combine_masks(associated_masks)


# given a list of individual nuclei masks, combine them into a 
# single mask image
def combine_masks(mask_list):
	mask = mask_list[0].im
	for m in mask_list:
		mask += m.im

	mask[mask > 0] = 1
	mask[mask <0]=0
	return mask


# Breaks up an image and mask pari into sub images/masks.
# This is super helpful to help normalize the data size, as 
# well as to create more data.
def convolve(image,mask,dim=128,sample_size=100):
	a = rgb2gray(image.im)
	sub_shape = (dim, dim)
	view_shape = tuple(np.subtract(a.shape, sub_shape) + 1) + sub_shape
	strides = a.strides + a.strides
	#strides = [10,11,12,13]
	sub_matrices = np.lib.stride_tricks.as_strided(a,view_shape,strides)


	# reshape
	#num_images = sub_matrices.shape[0]*sub_matrices.shape[1]
	#sub_matrices = sub_matrices.reshape(num_images,dim,dim);

	am = mask
	sub_shapem = (dim, dim)
	view_shapem = tuple(np.subtract(mask.shape, sub_shapem) + 1) + sub_shapem
	stridesm = am.strides + am.strides
	sub_matricesm = np.lib.stride_tricks.as_strided(am,view_shapem,stridesm)

	# reshape
	rand_row_indices = np.random.randint(len(sub_matricesm), size = sample_size)
	rand_col_indices = np.random.randint(len(sub_matricesm[0]), size = sample_size)

	sample_images = sub_matrices[rand_row_indices, rand_col_indices]
	sample_masks = sub_matricesm[rand_row_indices, rand_col_indices]

	return sample_images, sample_masks


### PIXEL ENCODING ### 
# source:
# Sam Stainsby 
# Fast,tested RLE and input routines
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



def rle_decode(rle_str, mask_shape, mask_dtype):
    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(np.prod(mask_shape), dtype=mask_dtype)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape(mask_shape[::-1]).T


### Plotting ###
def plot_confusion_matrix(array):
    df_cm = pd.DataFrame(array, index = [i for i in ["True Nuclei"," True Not-Nuclei"]],
                      columns = [i for i in ["Predicted Nuclei","Predicted Not-Nuclei"]])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
