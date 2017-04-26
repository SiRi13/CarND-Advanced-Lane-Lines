import pickle
import cv2
import os
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import advanced_lane_finder.constants as consts
%matplotlib inline

# %% load settings
settings_pickle_path = os.path.join(consts.SETTINGS_FOLDER, consts.SETTINGS_PICKLE)
settings_pickle = { consts.KEY_TIME_STAMP_CALIBRATION: time.ctime() }
# pickle.dump(settings_pickle, open(settings_pickle_path, mode='wb'))

settings_pickle = pickle.load(open(settings_pickle_path, mode='rb'))

# %% define functions
def undistort(img):
    return cv2.undistort(img, settings_pickle[consts.KEY_CALIBRATION_MATRIX], settings_pickle[consts.KEY_DISTORTION_COEFFICIENT])

def warp(img, img_size=consts.WARPED_SIZE, flags=cv2.WARP_FILL_OUTLIERS+cv2.INTER_CUBIC):
    return cv2.warpPerspective(img, settings_pickle[consts.KEY_TRANSFORMATION_MATRIX], img_size, flags=flags)

def blur(img):
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # plt.imshow(np.hstack((img_hls, img_lab)), cmap='hot')
    return cv2.medianBlur(img_hls, 5), cv2.medianBlur(img_lab, 5)

def get_biggest_contour(contours):
    biggest_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > biggest_area:
            biggest_area = area
            biggest_contour = contour

    return biggest_contour

def morphology_transformation(img, total_mask, line_mask, roi_mask, mask):
    # undistort, unwarp, change space, filter
    img = mpimg.imread('./test_images/test1.jpg')
    img = mpimg.imread('./test_images/straight_lines1.jpg')

    img = undistort(img)
    img = warp(img)
    hls, lab = blur(img)
    plt.imshow(lab[:, :, 2], cmap='hot')

    # get scenery
    noise = (np.uint8(lab[:, :, 2]) > 130) & cv2.inRange(hls, (0, 0, 50), (35, 190, 255))
    plt.imshow((np.uint8(lab[:, :, 2]) > 130), cmap='gray')
    plt.imshow(cv2.inRange(hls, (0, 0, 50), (35, 190, 255)), cmap='gray')
    noise[:, 100:200] = 0
    plt.imshow(noise, cmap='gray')

    # get road
    road_mask = np.uint8(np.logical_not(noise)) & (hls[:, :, 1] < 250)
    plt.imshow(road_mask, cmap='gray')
    # apply morphEx OPEN on road with 7x7 kernel
    kernel_7_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    morphEx = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel_7_rect)
    plt.imshow(morphEx, cmap='gray')
    # dilate morphed road with 31x31 kernel
    kernel_31_elli = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29))
    dilation = cv2.dilate(morphEx, kernel_31_elli)
    plt.imshow(dilation, cmap='gray')
    # get contours
    img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # get biggest contour/area
    max_contour = get_biggest_contour(contours)
    print('contours')
    tmp = cv2.drawContours(dilation, contours, -1, (255, 0, 0), thickness=5)
    plt.imshow(tmp, cmap='gray')
    # fill biggest contour on road mask
    road_mask = np.zeros_like(road_mask)
    road_mask = cv2.fillPoly(road_mask, [max_contour],  1)
    plt.imshow(road_mask, cmap='gray')
    # merge line with road
    roi_mask[:, :, 0] = line_mask & road_mask
    roi_mask[:, :, 1] = roi_mask[:, :, 0]
    roi_mask[:, :, 2] = roi_mask[:, :, 0]

    plt.imshow(roi_mask[:, :, 2], cmap='gray')

    # morphEx on lab and hls
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 3))
    black = cv2.morphologyEx(lab[:, :, 0], cv2.MORPH_TOPHAT, kernel)
    lines = cv2.morphologyEx(hls[:, :, 1], cv2.MORPH_TOPHAT, kernel)

    # plt.imshow(np.hstack((black, lines)))

    # apply morphEx with 13x13 kernel for yllw lines
    kernel_13_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    lines_yellow = cv2.morphologyEx(lab[:, :, 2], cv2.MORPH_TOPHAT, kernel_13_rect)

    # threshold maks
    mask[:, :, 0] = cv2.adaptiveThreshold(black, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, -6)
    mask[:, :, 1] = cv2.adaptiveThreshold(lines, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, -4)
    mask[:, :, 2] = cv2.adaptiveThreshold(lines_yellow, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, -1.5)
    # merge masks and erode
    mask *= roi_mask
    total_mask = np.uint8(np.any(mask, axis=2))
    kernel_3_elli = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    total_mask = cv2.morphologyEx(total_mask, cv2.MORPH_ERODE, kernel_3_elli)

    return total_mask

# %% run all test-mages
for img_path in glob.glob('./test_images/*jpg'):
    # %% load img
    img = mpimg.imread('./test_images/test1.jpg')
    img = mpimg.imread('./test_images/test1.jpg')
    # plt.imshow(img)

    # %% set up masks
    mask = np.zeros((consts.WARPED_SIZE[1], consts.WARPED_SIZE[0], 3), dtype=np.uint8)
    roi_mask = np.ones((consts.WARPED_SIZE[1], consts.WARPED_SIZE[0], 3), dtype=np.uint8)
    total_mask = np.zeros((consts.WARPED_SIZE[1], consts.WARPED_SIZE[0], 3), dtype=np.uint8)
    warped_mask = np.zeros((consts.WARPED_SIZE[1], consts.WARPED_SIZE[0]), dtype=np.uint8)
    line_mask = np.ones((consts.WARPED_SIZE[1], consts.WARPED_SIZE[0]), dtype=np.uint8)

    # %% run morph transformation
    total_mask = morphology_transformation(img, total_mask, line_mask, roi_mask, mask)
    # plt.imshow(total_mask, cmap='gray')
    mpimg.imsave(os.path.join('./debug_images/playground', os.path.basename(img_path).replace('jpg', 'png')), total_mask, cmap='gray')

    break

# %% convolutional search
%matplotlib qt
warped = cv2.imread('./debug_images/playground/test3.png', cv2.IMREAD_GRAYSCALE)
plt.imshow(warped)
# warped = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
n_windows = 11
window_height = warped.shape[0] // n_windows
window_width = 15
margin = 10

warped.shape[0] // window_height
test = warped[int(3 * warped.shape[0] / 4):, :int(warped.shape[1] / 2)]
test.shape

# %% define functions
def window_mask(width, height, img_ref, center, level):
    retVal = np.zeros_like(img_ref)
    retVal[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - (level * height)),
            max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1

    return retVal

def find_window_centroids(image, win_width, win_height, margin, left:True):
    # centroids positions per level (left, right)
    window_centroids = list()
    # window template for convolutions
    conv_a = np.ones(win_width)

    # 1. find starting positions for left/right lanes
    if left:
        conv_v = np.sum(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
        center = np.argmax(np.convolve(conv_a, conv_v)) - win_width / 2
    if not left:
        conv_v = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
        center = np.argmax(np.convolve(conv_a, conv_v)) - win_width / 2 + int(image.shape[1] / 2)

    # add as first layer
    window_centroids.append(center)

    # iterate over layers looking for max pixel locations
    for level in range(1, n_windows):
        # convolve window into vertical slice of image
        conv_v = np.sum(image[int(image.shape[0] - (level + 1) * win_height):int(image.shape[0] - (level * win_height)), :], axis=0)
        conv_signal = np.convolve(conv_a, conv_v)
        # find best left centroid by using past centroids as a reference
        # conv_signal is the right side of window => win_width/2
        offset = win_width / 2
        if left:
            l_min_idx = int(max(center + offset - margin, 0))
            l_max_idx = int(min(center + offset + margin, image.shape[1]))
            center = np.argmax(conv_signal[l_min_idx:l_max_idx]) + l_min_idx - offset
        if not left:
            # find best right centroid with past centroid as reference
            r_min_idx = int(max(center + offset - margin, 0))
            r_max_idx = int(min(center + offset + margin, image.shape[1]))
            center = np.argmax(conv_signal[r_min_idx:r_max_idx]) + r_min_idx - offset
        # add to list
        window_centroids.append(center)

    return window_centroids

# %% run search
window_centroids_left = find_window_centroids(warped, window_width, window_height, margin, left=True)
window_centroids_right = find_window_centroids(warped, window_width, window_height, margin, left=False)

# %% if there are any centroids
# points to draw left/right windows
l_pts = np.zeros_like(warped)
r_pts = np.zeros_like(warped)

# iterate levels and draw windows
for level in range(len(window_centroids_left)):
    # draw window areas
    l_mask = window_mask(window_width, window_height, warped, window_centroids_left[level], level)
    r_mask = window_mask(window_width, window_height, warped, window_centroids_right[level], level)
    # add graphic points to masks
    l_pts[(l_pts == 255) | (l_mask == 1)] = 255
    r_pts[(r_pts == 255) | (r_mask == 1)] = 255

# add left and right window pixels together
template = np.array(r_pts + l_pts, np.uint8)
# create zero color channel
zero_chan = np.zeros_like(l_pts)
# make window pixels green (R, G, B)
template = np.array(cv2.merge((l_pts, zero_chan, r_pts)), np.uint8)
# make original road pixels 3 channels deep
warpage = np.array(cv2.merge((warped, warped, warped)), np.uint8)
# overlay original road image with window results in template
output = cv2.addWeighted(warpage, 1, template, 0.7, 0.0)

# plt.imshow(warped)
# %matplotlib qt
plt.imshow(output)
# plt.show()

def fit_line(warped, points):
    y, x = np.where(points)

    fit = np.polyfit(y, x, 2)

    plot_y = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    fit_x = fit[0] * plot_y**2 + fit[1] * plot_y + fit[2]

    # plot output_image
    plt.figure(figsize=(15,10))
    plt.imshow(warped)
    plt.plot(fit_x, plot_y, 'r+')
    plt.xlim(0, warped.shape[1])
    plt.ylim(warped.shape[0], 0)
    plt.show()

fit_line(warped, l_pts)
fit_line(warped, r_pts)

# %% test
warped = undist_warp(img)
hls, lab = blur(warped)

kernel_13_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
lines_yellow = cv2.morphologyEx(lab[:, :, 2], cv2.MORPH_TOPHAT, kernel_13_rect)
plt.imshow(lines_yellow)


yw_mask = cv2.inRange(lines_yellow, 7, 80)
res = cv2.bitwise_or(mask, mask, mask=yw_mask)
plt.imshow(res, cmap='hot')


img = mpimg.imread('./test_images/test4.jpg')
img = cv2.undistort(img, settings_pickle[consts.KEY_CALIBRATION_MATRIX], settings_pickle[consts.KEY_DISTORTION_COEFFICIENT])
src_coords = np.float32([[240,719],
                         [579,450],
                         [712,450],
                         [1165,719]])
dst_coords = np.float32([[300,719],
                         [300,0],
                         [900,0],
                         [900,719]])

img = cv2.polylines(img, [src_coords.astype(np.int32)], True, (255, 0, 0), thickness=4)
img = cv2.polylines(img, [dst_coords.astype(np.int32)], True, (255, 0, 0), thickness=4)
plt.imshow(img)

# %% hist peak search
warped = cv2.imread('./debug_images/playground/test5.png', cv2.IMREAD_GRAYSCALE)
plt.imshow(warped, cmap='gray')

histogram = np.sum(warped[150:, :], axis=0)

# %% create output image
output_image = np.dstack((warped, warped, warped))*255
# find peaks
midpoint = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

# %% choose sliding windows
n_slid_windows = 18
# window height
window_height = np.int(warped.shape[0]/n_slid_windows)
non_zero = warped.nonzero()
non_zero_y = np.array(non_zero[0])
non_zero_x = np.array(non_zero[1])

# %% current position
leftx_current = leftx_base
rightx_current = rightx_base
# set width of windows margin
margin = 30
# min number of pxls to recenter sliding window
min_pix = 30
# lists for lane pixel idcs
left_lane_indcs, right_lane_indcs = list(), list()

# %% step through windows
for window in range(n_slid_windows):
    # identify boundries in x and y
    win_y_low = warped.shape[0] - (window + 1) * window_height
    win_y_high = warped.shape[0] - window * window_height
    wind_xleft_low = leftx_current - margin
    wind_xleft_high = leftx_current + margin
    wind_xright_low = rightx_current - margin
    wind_xright_high = rightx_current + margin
    # draw visualization of window
    cv2.rectangle(output_image, (wind_xleft_low, win_y_low), (wind_xleft_high, win_y_high), (0, 255, 0), 2)
    cv2.rectangle(output_image, (wind_xright_low, win_y_low), (wind_xright_high, win_y_high), (0, 255, 0), 2)
    # identify nonzero pixels in x an y
    good_left_indcs = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high)
                        & (non_zero_x >= wind_xleft_low) & (non_zero_x < wind_xleft_high)).nonzero()[0]
    good_right_indcs = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high)
                        & (non_zero_x >= wind_xright_low) & (non_zero_x < wind_xright_high)).nonzero()[0]
    # append to lists
    left_lane_indcs.append(good_left_indcs)
    right_lane_indcs.append(good_right_indcs)
    # if > min_pix, recenter
    if len(good_left_indcs) > min_pix:
        leftx_current = np.int(np.mean(non_zero_x[good_left_indcs]))
    if len(good_right_indcs) > min_pix:
        rightx_current = np.int(np.mean(non_zero_x[good_right_indcs]))

# %% concatenate list of indices
left_lane_indcs = np.concatenate(left_lane_indcs)
right_lane_indcs = np.concatenate(right_lane_indcs)

# extract left and right line pixel positions
left_x = non_zero_x[left_lane_indcs]
left_y = non_zero_y[left_lane_indcs]
right_x = non_zero_x[right_lane_indcs]
right_y = non_zero_y[right_lane_indcs]

# fit 2nd order polynom
left_fit = np.polyfit(left_y, left_x, 2)
right_fit = np.polyfit(right_y, right_x, 2)

# %% visualize windows
plot_y = np.linspace(0, warped.shape[0]-1, warped.shape[0])
left_fit_x = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
right_fit_x = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]

# color found pixels
output_image[non_zero_y[left_lane_indcs], non_zero_x[left_lane_indcs]] = [255, 0, 0]
output_image[non_zero_y[right_lane_indcs], non_zero_x[right_lane_indcs]] = [0, 0, 255]

# plot output_image
plt.figure(figsize=(15,25))
plt.imshow(output_image)
plt.plot(left_fit_x, plot_y, color='yellow')
plt.plot(right_fit_x, plot_y, color='yellow')
plt.xlim(0, warped.shape[1])
plt.ylim(warped.shape[0], 0)
plt.show()

# %% skip sliding window search for following frames
next_warped = cv2.imread('./debug_images/playground/test5.png', cv2.IMREAD_GRAYSCALE)
plt.imshow(next_warped, cmap='gray')

non_zero = next_warped.nonzero()
non_zero_y = np.array(non_zero[0])
non_zero_x = np.array(non_zero[1])
margin = 25


# %% extract left pixel positions again
left_lane_indcs = ((non_zero_x > (left_fit[0]*(non_zero_y**2) + left_fit[1]*non_zero_y + left_fit[2] - margin))
                    & (non_zero_x < (left_fit[0]*(non_zero_y**2) + left_fit[1]*non_zero_y + left_fit[2] + margin)))

left_x = non_zero_x[left_lane_indcs]
left_y = non_zero_y[left_lane_indcs]

plt.imshow(next_warped, cmap='gray')
plt.scatter(left_x, left_y, marker='o', cmap='gray')
plt.show()

# %% right pixel positions again
right_lane_indcs = ((non_zero_x > (right_fit[0] * (non_zero_y**2) + right_fit[1] * non_zero_y + right_fit[2] - margin))
                    & (non_zero_x < (right_fit[0] * (non_zero_y**2) + right_fit[1] * non_zero_y + right_fit[2] + margin)))
right_x = non_zero_x[right_lane_indcs]
right_y = non_zero_y[right_lane_indcs]


# %% fit polynom
left_fit = np.polyfit(left_y, left_x, 2)
right_fit = np.polyfit(right_y, right_x, 2)
# generate x/y values for plotting
plot_y = np.linspace(0, next_warped.shape[0]-1, next_warped.shape[0])
left_fit_x = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
right_fit_x = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]

# %% visualize it again
output_image_2 = np.dstack((next_warped, next_warped, next_warped)) * 255
win_image = np.zeros_like(output_image_2)
# color line pixels
output_image_2[non_zero_y[left_lane_indcs], non_zero_x[left_lane_indcs]] = [255, 0, 0]
output_image_2[non_zero_y[right_lane_indcs], non_zero_x[right_lane_indcs]] = [0, 0, 255]

# plot output_image
plt.figure(figsize=(15,25))
plt.imshow(output_image_2)
plt.plot(left_fit_x, plot_y, color='yellow')
plt.plot(right_fit_x, plot_y, color='yellow')
plt.xlim(0, next_warped.shape[1])
plt.ylim(next_warped.shape[0], 0)
plt.show()

# %% generate polygon
left_line_window_1 = np.array([np.transpose(np.vstack([left_fit_x - margin, plot_y]))])
left_line_window_2 = np.array([np.flipud(np.transpose(np.vstack([left_fit_x + margin, plot_y])))])
left_line_points = np.hstack((left_line_window_1, left_line_window_2))
right_line_window_1 = np.array([np.transpose(np.vstack([right_fit_x - margin, plot_y]))])
right_line_window_2 = np.array([np.flipud(np.transpose(np.vstack([right_fit_x + margin, plot_y])))])
right_line_points = np.hstack((right_line_window_1, right_line_window_2))

# %% draw lane onto image
cv2.fillPoly(win_image, np.int_([left_line_points]), (0, 255, 0))
cv2.fillPoly(win_image, np.int_([right_line_points]), (0, 255, 0))
result = cv2.addWeighted(output_image_2, 1, win_image, 0.3, 0)

# %% plot result
plt.figure(figsize=(15,10))
plt.imshow(result)
plt.plot(left_fit_x, plot_y, color='yellow')
plt.plot(right_fit_x, plot_y, color='yellow')
plt.xlim(0, next_warped.shape[1])
plt.ylim(next_warped.shape[0], 0)
plt.show()
