import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    if orient == 'y':
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel = np.abs(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Apply threshold
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelxy = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    # Apply threshold
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.sqrt(sobelx**2)
    abs_sobely = np.sqrt(sobely**2)
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    # Apply threshold
    dir_binary = np.zeros_like(grad_dir)
    dir_binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    return dir_binary

# Choose a Sobel kernel size
ksize = 15 # Choose a larger odd number to smooth gradient measurements

# load image
org_image = mpimg.imread('./images/signs_vehicles_xygrad.png')
image = cv2.cvtColor(np.copy(org_image), cv2.COLOR_RGB2GRAY)

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))

all_combined = np.zeros_like(dir_binary)
all_combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

mag_dir_combined = np.zeros_like(dir_binary)
mag_dir_combined[(dir_binary == 1) & (mag_binary ==1)] = 1

grad_dir_combined = np.zeros_like(dir_binary)
grad_dir_combined[((gradx == 1) & (grady == 1) | (dir_binary == 1))] = 1

grad_mag_combined = np.zeros_like(mag_binary)
grad_mag_combined[((gradx == 1) & (grady == 1) | (mag_binary == 1))] = 1

# plot results
f, axN = plt.subplots(5, 1, figsize=(15, 15))
axN[0].imshow(org_image)
axN[0].set_title('Original Image', fontsize=15)
axN[1].imshow(all_combined, cmap='gray')
axN[1].set_title('All Combined', fontsize=15)
axN[2].imshow(mag_dir_combined, cmap='gray')
axN[2].set_title('Gradient Magnitude & Direction Combined', fontsize=15)
axN[3].imshow(grad_dir_combined, cmap='gray')
axN[3].set_title('Directional Gradient & Gradient Direction Combined', fontsize=15)
axN[4].imshow(mag_dir_combined, cmap='gray')
axN[4].set_title('Gradient Magnitude & Directional Gradient Combined', fontsize=15)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
f.tight_layout()
plt.show()
