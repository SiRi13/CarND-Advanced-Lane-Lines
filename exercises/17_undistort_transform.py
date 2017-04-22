import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# %% import mtx and dist from pickle
dist_pickle = pickle.load(open('../camera_cal/dist_pickle.p', 'rb'))
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']

# %% load image
fname = '../camera_cal/calibration2.jpg'
test_img = mpimg.imread(fname)
nx = 9
ny = 6

def corners_unwrap(img, nx, ny, mtx, dist):
    # 1. undistort
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # 2. convert to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)
    # 3. find corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # 4. corners were found
    if ret == True:
        offset = 100
        img_size = (gray.shape[1], gray.shape[0])
        # a) draw corners
        undist = cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        # b) define 4 source points
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        # c) define 4 destination points
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
                          [img_size[0]-offset, img_size[1]-offset], [offset, img_size[1]-offset]])
        # d) generate transform matrix M
        M = cv2.getPerspectiveTransform(src, dst)
        # e) warp image to top-down view
        warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)
    else:
        warped = undist
        M = None

    return warped, M

# %% unwrap and save image
top_down, perspective_M = corners_unwrap(test_img, nx, ny, mtx, dist)
if perspective_M is not None:
    cv2.imwrite('./images/calibration2_warped.jpg', top_down)

# %% plot inline
%matplotlib inline
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,10))
f.tight_layout()
ax1.imshow(test_img)
ax1.set_title('Original Image', fontsize=20)
ax1.imshow(top_down)
ax1.set_title('Undistorted and Warped Image', fontsize=20)
plt.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.0)
plt.show()
