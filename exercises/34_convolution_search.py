import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2

# %% load image
warped = mpimg.imread('./images/warped_example.jpg')
# window settings
n_windows = 9
window_height = warped.shape[0] // n_windows
window_width = 50
margin = 100

def window_mask(width, height, img_ref, center, level):
    retVal = np.zeros_like(img_ref)
    retVal[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - (level * height)),
            max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1

    return retVal

def find_window_centroids(image, win_width, win_height, margin):
    # centroids positions per level (left, right)
    window_centroids = list()
    # window template for convolutions
    win = np.ones(win_width)

    # 1. find starting positions for left/right lanes
    l_sum = np.sum(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(win, l_sum)) - win_width / 2
    r_sum = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(win, r_sum)) - win_width / 2 + int(image.shape[1] / 2)

    # add as first layer
    window_centroids.append((l_center, r_center))

    # iterate over layers looking for max pixel locations
    for level in range(1, int(image.shape[0] / win_height)):
        # convolve window into vertical slice of image
        img_layer = np.sum(image[int(image.shape[0] - (level + 1) * win_height):int(image.shape[0] - (level * win_height)), :], axis=0)
        conv_signal = np.convolve(win, img_layer)
        # find best left centroid by using past centroids as a reference
        # conv_signal is the right side of window => win_width/2
        offset = win_width / 2
        l_min_idx = int(max(l_center + offset - margin, 0))
        l_max_idx = int(min(l_center + offset + margin, image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_idx:l_max_idx]) + l_min_idx - offset
        # find best right centroid with past centroid as reference
        r_min_idx = int(max(r_center + offset - margin, 0))
        r_max_idx = int(min(r_center + offset + margin, image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_idx:r_max_idx]) + r_min_idx - offset
        # add to list
        window_centroids.append((l_center, r_center))

    return window_centroids

window_centroids = find_window_centroids(warped, window_width, window_height, margin)

# %% if there are any centroids
if len(window_centroids) > 0:
    # points to draw left/right windows
    l_pts = np.zeros_like(warped)
    r_pts = np.zeros_like(warped)

    # iterate levels and draw windows
    for level in range(len(window_centroids)):
        # draw window areas
        if level <= 1:
            l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
        # add graphic points to masks
        l_pts[(l_pts == 255) | (l_mask == 1)] = 255
        r_pts[(r_pts == 255) | (r_mask == 1)] = 255

    # add left and right window pixels together
    template = np.array(r_pts + l_pts, np.uint8)
    # create zero color channel
    zero_chan = np.zeros_like(template)
    # make window pixels green (R, G, B)
    template = np.array(cv2.merge((zero_chan, template, zero_chan)), np.uint8)
    # make original road pixels 3 channels deep
    warpage = np.array(cv2.merge((warped, warped, warped)), np.uint8)
    # overlay original road image with window results in template
    output = cv2.addWeighted(warpage, 1, template, 0.7, 0.0)
else:
    # no centroids found, display road image
    output = np.array((warped, warped, warped), np.uint8)

plt.figure(figsize=(15, 10))
plt.imshow(output)
plt.title('convolution results')
plt.show()
