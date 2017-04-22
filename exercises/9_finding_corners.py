import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# %% object points
nx = 9
ny = 6

# %% load image
fname = '../camera_cal/calibration2.jpg'
cal_img = mpimg.imread(fname)

# %% show image
plt.imshow(cal_img)
plt.show()

# %% convert to grayscale
gray_cal_img = cv2.cvtColor(cal_img, cv2.COLOR_RGB2GRAY)

# %% find chess board corners
ret, corners = cv2.findChessboardCorners(gray_cal_img, (nx, ny), None)

# %% if found display on image
if ret == True:
    print("found corners")
    cal_img = cv2.drawChessboardCorners(cal_img, (nx, ny), corners, ret)

plt.imshow(cal_img)
plt.show()
