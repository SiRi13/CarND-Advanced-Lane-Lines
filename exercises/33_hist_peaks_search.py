import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# %% load warped image & convert to gray
warped = mpimg.imread('./images/warped_example.jpg')
if warped.shape[-1] == 3:
    warped = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)

plt.figure(figsize=(15,15))
plt.imshow(warped, cmap='gray')
plt.show()

# %% create histogram for bottom of image
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
margin = 100
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
plt.figure(figsize=(15,10))
plt.imshow(output_image)
plt.plot(left_fit_x, plot_y, color='yellow')
plt.plot(right_fit_x, plot_y, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.show()

# %% skip sliding window search for following frames
next_warped = mpimg.imread('./images/warped_example2.jpg')
non_zero = next_warped.nonzero()
non_zero_x = np.array(non_zero[0])
non_zero_y = np.array(non_zero[1])
margin = 100
left_lane_indcs = ((non_zero_x > (left_fit[0] * (non_zero_y**2) + left_fit[1] * non_zero_y + left_fit[2] - margin))
                    & (non_zero_x < (left_fit[0] * (non_zero_y**2) + left_fit[1] * non_zero_y + left_fit[2] + margin)))
right_lane_indcs = ((non_zero_x > (right_fit[0] * (non_zero_y**2) + right_fit[1] * non_zero_y + right_fit[2] - margin))
                    & (non_zero_x < (right_fit[0] * (non_zero_y**2) + right_fit[1] * non_zero_y + right_fit[2] + margin)))

# %% extract left/right pixel positions again
left_x = non_zero_x[left_lane_indcs]
left_y = non_zero_y[left_lane_indcs]
right_x = non_zero_x[right_lane_indcs]
right_y = non_zero_y[right_lane_indcs]
# fit polynom
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
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.show()
