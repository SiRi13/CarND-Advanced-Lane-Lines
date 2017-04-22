import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

# %% import dist mtx
dist_pickle = pickle.load(open('../camera_cal/dist_pickle.p', 'rb'))
obj_points = dist_pickle['objpoints']
img_points = dist_pickle['imgpoints']

# %% load image
fname = '../camera_cal/calibration2.jpg'
cal_img = mpimg.imread(fname)

# %% show image
plt.imshow(cal_img)
plt.show()

def cal_undistort(img, objpoints, imgpoints):
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    dest = cv2.undistort(img, mtx, dist, None, mtx)
    return dest

# %% correct image and save
destination_img = cal_undistort(cal_img, obj_points, img_points)
cv2.imwrite('./images/calibration2_undist.jpg', destination_img)

# %% plot images
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(cal_img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(destination_img)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
