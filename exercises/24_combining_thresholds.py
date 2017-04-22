from helper import plot_1_2
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import importlib
apply_sobel = importlib.import_module("21_apply_sobel")
mag_thresh = importlib.import_module("22_mag_thresh")
dir_grad = importlib.import_module("23_dir_grad")

# Choose a Sobel kernel size
ksize = 15 # Choose a larger odd number to smooth gradient measurements

# %% load image and convert to grayscale
image = mpimg.imread('./images/signs_vehicles_xygrad.png')
gray = cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2GRAY)

# %% Apply each of the thresholding functions
gradx = apply_sobel.abs_sobel_thresh(gray, 'x', sobel_kernel=ksize, threshold=(20, 100))
grady = apply_sobel.abs_sobel_thresh(gray, 'y', sobel_kernel=ksize, threshold=(20, 100))
mag_binary = mag_thresh.mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(30, 100))
dir_binary = dir_grad.dir_thresh(gray, sobel_kernel=ksize, threshold=(0.7, 1.3))

all_combined = np.zeros_like(dir_binary)
all_combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

mag_dir_combined = np.zeros_like(dir_binary)
mag_dir_combined[(dir_binary == 1) & (mag_binary ==1)] = 1

grad_dir_combined = np.zeros_like(dir_binary)
grad_dir_combined[((gradx == 1) & (grady == 1) | (dir_binary == 1))] = 1

grad_mag_combined = np.zeros_like(mag_binary)
grad_mag_combined[((gradx == 1) & (grady == 1) | (mag_binary == 1))] = 1

# %% plot results
plot_1_2((image, all_combined), ('Original Image', 'All Combined'), grayscale=(None, 'gray'))
plot_1_2((mag_dir_combined, grad_dir_combined), ('Gradient Magnitude & Direction Combined', 'Directional Gradient & Gradient Direction Combined'), grayscale=('gray', 'gray'))
plot_1_2((image, mag_dir_combined), ('Original Image', 'Gradient Magnitude & Directional Gradient Combined'), grayscale=(None, 'gray'))
