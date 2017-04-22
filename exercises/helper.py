import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_1_2(images, titles, grayscale=(None, None)):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,10))
    f.tight_layout()
    ax1.imshow(images[0], cmap=grayscale[0])
    ax1.set_title(titles[0], fontsize=20)
    ax2.imshow(images[1], cmap=grayscale[1])
    ax2.set_title(titles[1], fontsize=20)
    plt.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.0)
    plt.show()
