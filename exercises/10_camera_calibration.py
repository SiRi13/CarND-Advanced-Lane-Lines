import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

# open designated matplot window
%matplotlib qt

# %% object points
nx = 9
ny = 6

# %% load images
fnames = '../camera_cal/calibration*.jpg'
cal_images = glob.glob(fnames)

# %% arrays for object and image points
# 3d points in real world
obj_points = []
# 2d points on image plane (corners)
img_points = []
# object point template
objp = np.zeros((ny * nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# %% step through images and search for corners
for idx, fname in enumerate(cal_images):
    # load image
    cal_img = mpimg.imread(fname)
    # convert to grayscale
    gray_cal_img = cv2.cvtColor(cal_img, cv2.COLOR_RGB2GRAY)
    # find chess board corners
    ret, corners = cv2.findChessboardCorners(gray_cal_img, (nx, ny), None)
    # if successfull
    if ret == True:
        img_points.append(corners)
        obj_points.append(objp)

        img = cv2.drawChessboardCorners(cal_img, (nx, ny), corners, ret)
        cv2.imshow('cal img with corners', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# test undistortion
test_img = mpimg.imread('../camera_cal/calibration2.jpg')
test_img_size = (test_img.shape[1], test_img.shape[0])

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, test_img_size, None, None)

dest = cv2.undistort(test_img, mtx, dist, None, mtx)
cv2.imwrite('./images/calibration2_undist.jpg', dest)

# %% export calibration as pickle
dist_pickle = dict()
dist_pickle['objpoints'] = obj_points
dist_pickle['imgpoints'] = img_points
dist_pickle['mtx'] = mtx
dist_pickle['dist'] = dist
pickle.dump(dist_pickle, open('../camera_cal/dist_pickle.p', 'wb'))
