import numpy as np
import matplotlib.pyplot as plt

# %% generate fake lane-line pixels
# cover same width
plot_y = np.linspace(0, 719, num=720)
# arbitrary coefficient
quadratic_coeff = 3e-4
# generate random x for each y in range of +/- 50 at x=200 and x=900
left_x = np.array([200 + (y**2) * quadratic_coeff + np.random.randint(-50, 51) for y in plot_y])
right_x = np.array([900 + (y**2) * quadratic_coeff + np.random.randint(-50, 51) for y in plot_y])

# reverse to match top-to-bottom in y
left_x = left_x[::-1]
right_x = right_x[::-1]

# %% fit polynom to positions
left_fit = np.polyfit(plot_y, left_x, 2)
left_fit_x = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
right_fit = np.polyfit(plot_y, right_x, 2)
right_fit_x = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]

# %% plot fake data
mark_size = 3
plt.figure(figsize=(15, 10))
plt.plot(left_x, plot_y, 'o', color='red', markersize=mark_size)
plt.plot(right_x, plot_y, 'o', color='blue', markersize=mark_size)
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.plot(left_fit_x, plot_y, color='green', linewidth=3)
plt.plot(right_fit_x, plot_y, color='green', linewidth=3)
plt.gca().invert_yaxis()
plt.show()

# %% define y-value where we want the radius
# max of plot_y ^= bottom of image
y_eval = np.max(plot_y)
left_line_curve_rad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1])**2)**1.5) / np.absolute(2 * left_fit[0])
right_line_curve_rad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1])**2)**1.5) / np.absolute(2 * right_fit[0])
# define conversion from pixel space to meters
ym_per_pix = 30/720
xm_per_pix = 3.7/700
# fit new polynom to x/y in world space
left_fit_cr = np.polyfit(plot_y * ym_per_pix, left_x * xm_per_pix, 2)
right_fit_cr = np.polyfit(plot_y * ym_per_pix, right_x * xm_per_pix, 2)
# calculate new radii of curvature
left_curve_rad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
right_curve_rad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])
# now it is in meters
print(left_curve_rad, 'm', right_curve_rad, 'm')
