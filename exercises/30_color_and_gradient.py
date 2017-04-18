import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def pipeline(img, v_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # convert to HLS color space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hsv[:,:,1]
    v_channel = hsv[:,:,2]

    # sobel x
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # threshold sobel x
    sxbinary = np.zeros_like(scaled_sobelx)
    sxbinary[(scaled_sobelx >= sx_thresh[0]) & (scaled_sobelx <= sx_thresh[1])] = 1

    # threshold color channel
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1

    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, v_binary))
    return color_binary

image = mpimg.imread('./images/test4.jpg')
result = pipeline(image)

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 9))
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=40)
ax2.imshow(result)
ax2.set_title('Pipeline Result', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
f.tight_layout()
plt.show()
