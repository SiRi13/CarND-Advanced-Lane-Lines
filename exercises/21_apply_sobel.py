import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import helper

# %% sobel threshold function
def abs_sobel_thresh(img, orient='x', threshold=(0, 255)):
    # 1. take derivate in x or y depending on orient
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    if orient == 'y':
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    # 2. get absolute
    abs_sobel = np.absolute(sobel)
    # 3. scale 8-bit and convert to np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 4. create mask
    mask = np.zeros_like(scaled_sobel)
    # 5. apply threshol
    mask[(scaled_sobel > threshold[0]) & (scaled_sobel < threshold[1])] = 1
    # 6. return binary mask
    return mask

# %% load image and grayscale it
image = mpimg.imread('./images/signs_vehicles_xygrad.png')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# %% run function
grad_binary = abs_sobel_thresh(gray, orient='x', threshold=(20, 100))

helper.plot_1_2((image, grad_binary), ('Original Image', 'Threshold Gradient'), (None, 'gray'))
