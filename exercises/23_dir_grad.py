import helper
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# %% define direction threshold function
def dir_thresh(img, sobel_kernel=3, threshold=(0, np.pi/2)):
    # 1. get gradient seperately
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 2. get absolute seperately
    abs_sobel_x = np.sqrt(sobel_x**2)
    abs_sobel_y = np.sqrt(sobel_y**2)
    # 3. calculate direction with arctan2
    grad_dir = np.arctan2(abs_sobel_y, abs_sobel_x)
    # 4. binary mask
    binary_mask = np.zeros_like(grad_dir)
    # 5. apply threshold
    binary_mask[(grad_dir > threshold[0]) & (grad_dir < threshold[1])] = 1
    # 6. return binary mask
    return binary_mask

# %% load image and convert to grayscale
image = mpimg.imread('./images/signs_vehicles_xygrad.png')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# %% apply function and plot
dir_binary = dir_thresh(gray, sobel_kernel=15, threshold=(0.7, 1.3))
helper.plot_1_2((image, dir_binary), ('Original Image', 'Thresholded Gradient Direction'), grayscale=(None, 'gray'))
