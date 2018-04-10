import numpy as np
import scipy
from tqdm import tqdm
from scipy import stats
import data_utils as du
import models
from unet import UNET
import warnings
import skimage
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
train_image_data = du.get_train_images()
train_mask_data  = du.get_train_masks()
test_image_data = du.get_test_images()


# preprocess
for x in tqdm(train_image_data):
    du.preprocess_image(x)
    
for x in tqdm(test_image_data):
    du.preprocess_image(x)

total_masks = []
for image in tqdm(train_image_data):
    Big_Mask = du.get_total_mask(image, train_mask_data)
    total_masks.append(Big_Mask)



IMG_WIDTH = 128
IMG_HEIGHT = 128
sub_images = np.empty((1,IMG_WIDTH,IMG_HEIGHT))
sub_masks = np.empty((1,IMG_WIDTH,IMG_HEIGHT))
samples_per_image = 100
num_images = 670

num_total = samples_per_image*num_images
sub_images = np.zeros((num_total,IMG_WIDTH,IMG_HEIGHT))
sub_masks = np.zeros((num_total,IMG_WIDTH,IMG_HEIGHT))

#for i in range(len(train_image_data)):
for i in tqdm(range(0,num_images)):
    sub_images_i, sub_masks_i = du.convolve(train_image_data[i], total_masks[i],dim=IMG_HEIGHT, sample_size=samples_per_image)
    sub_images[i*samples_per_image:(i+1)*samples_per_image] = sub_images_i
    sub_masks[i*samples_per_image:(i+1)*samples_per_image] = sub_masks_i
#     sub_images = np.append(sub_images, sub_images_i, axis=0)
#     print sub_images.shape
#     sub_masks = np.append(sub_masks, sub_masks_i, axis=0)


X = sub_images.reshape(len(sub_images), IMG_WIDTH, IMG_HEIGHT, 1)
y = sub_masks.reshape(len(sub_masks), IMG_WIDTH, IMG_HEIGHT, 1)

# for some reason the first X gets corrupted on sphere
X = X[1:]
y = y[1:]

# make sure our data is clean
# assert np.sum(np.isnan(X)) == 0
# assert np.sum(np.isnan(y)) == 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

unet = UNET(X_train)
unet.train(X_train, y_train, X_test, y_test)

plt.plot(unet.history.history['loss'])
plt.plot(unet.history.history['val_loss'])
plt.title('UNet Binary Cross-entropy loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train loss', 'Val loss'], loc='upper right')

plt.savefig("train_val_loss.png",dpi=500)


