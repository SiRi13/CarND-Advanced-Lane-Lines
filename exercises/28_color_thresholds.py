import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_results(images, titles):
    # plot results
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 15))
    ax1.imshow(images[0], cmap='gray')
    ax1.set_title(titles[0], fontsize=15)
    ax2.imshow(images[1], cmap='gray')
    ax2.set_title(titles[1], fontsize=15)
    ax3.imshow(images[2], cmap='gray')
    ax3.set_title(titles[2], fontsize=15)

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    f.tight_layout()
    plt.show()

def apply_threshold(img, thresh = (180, 255)):
    binary = np.zeros_like(img)
    binary[(img > thresh[0]) & (img <= thresh[1])] = 1
    return binary

image = mpimg.imread('./images/test6.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
binary = apply_threshold(gray)

plot_results([image, gray, binary], ['RGB', 'GRAY', 'GRAY Binary'])

plot_results([image[:,:,0], image[:,:,1], image[:,:,2]], ['R', 'G', 'B'])

R = image[:,:,0]
binary = apply_threshold(R, (200, 255))

plot_results([gray, R, binary], ['gray', 'R', 'R Binary'])

hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
H = hls[:,:,0]
L = hls[:,:,1]
S = hls[:,:,2]

plot_results([H, L, S], ['H', 'L', 'S'])

binary = apply_threshold(S, (90, 255))
plot_results([hls, S, binary], ['HLS', 'S', 'S Binary'])

binary = apply_threshold(H, (15, 100))
plot_results([hls, H, binary], ['HLS', 'H', 'H binary'])

image = mpimg.imread('./images/test4.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
binary = apply_threshold(gray)

plot_results([image, gray, binary], ['RGB', 'GRAY', 'GRAY Binary'])

R = image[:,:,0]
binary = apply_threshold(R, (200, 255))
plot_results([gray, R, binary], ['gray', 'R', 'R Binary'])

hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
H = hls[:,:,0]
L = hls[:,:,1]
S = hls[:,:,2]

plot_results([H, L, S], ['H', 'L', 'S'])

binary = apply_threshold(S, (90, 255))
plot_results([hls, S, binary], ['HLS', 'S', 'S Binary'])

binary = apply_threshold(H, (15, 100))
plot_results([hls, H, binary], ['HLS', 'H', 'H binary'])
