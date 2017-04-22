import helper
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# %% magnitude threshold function
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # 1. take x and y gradient seperately
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 2. calculate magnitude
    abs_sobelxy = np.sqrt(sobel_x**2 + sobel_y**2)
    # 3. scale to 8-bit and convert to uint8
    scaled_sobel = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))
    # 4. create binary mask
    binary_mask = np.zeros_like(scaled_sobel)
    # 5. apply threshold
    binary_mask[(scaled_sobel > mag_thresh[0]) & (scaled_sobel < mag_thresh[1])] = 1
    # 6. return mask
    return binary_mask

# %% load and convert image to gray
image = mpimg.imread('./images/signs_vehicles_xygrad.png')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# %% apply magnitude threshold
mag_binary = mag_thresh(gray, sobel_kernel=9, mag_thresh=(30, 100))
helper.plot_1_2((image, mag_binary), ('Original Image', 'Thresholded Magnitude'), (None, 'gray'))
