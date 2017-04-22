import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class BinaryImage:
    def __init__(self, img, kernel):
        self.image = img
        self.sobel_kernel = kernel

    def __Sobel_x(self, img, kernel=self.sobel_kernel):
        return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel)

    def __Sobel_y(self, img, kernel=self.sobel_kernel):
        return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel)

    def __scaled_uint8(self, abs_sobel):
        return np.uint8(255 * abs_sobel / np.max(abs_sobel))

    def __apply_threshold(self, mask, scaled_sobel, threshold):
        return mask[(scaled_sobel > threshold[0]) & (scaled_sobel < threshold[1])] = 1

    def absolute_threshold(self, orient='x', kernel=self.sobel_kernel, threshold=(20, 100)):
        # x/y derivative depending on orient
        sobel = self.__Sobel_x(img, kernel) if orient == 'x' else self.__Sobel_y(img, kernel)
        # get aboslute
        abs_sobel = np.absolute(sobel)
        # scale to 8-bit and convert to np.uint8
        scaled = self.__scaled_uint8(abs_sobel)
        # create binary mask
        binary_mask = np.zeros_like(scaled)
        # apply threshold
        binary_mask = self.__apply_threshold(binary_mask, scaled, threshold)
        # return binary mask
        return binary_mask

    def magnitude_threshold(self, kernel=self.sobel_kernel, threshold=(30, 100)):
        # get x and y derivative
        sobel_x = self.__Sobel_x(img, kernel)
        sobel_y = self.__Sobel_y(img, kernel)
        # get aboslute
        abs_sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        # scale to 8-bit and convert to np.uint8
        scaled = self.__scaled_uint8(abs_sobel)
        # create binary mask
        binary_mask = np.zeros_like(scaled)
        # apply threshold
        binary_mask = self.__apply_threshold(binary_mask, scaled, threshold)
        # return binary mask
        return binary_mask

    def direction_threshold(self, orient='x', kernel=self.sobel_kernel, threshold=(0.7, 1.3)):
        # get x and y derivative
        sobel_x = self.__Sobel_x(img, kernel)
        sobel_y = self.__Sobel_y(img, kernel)
        # get absolute seperately
        abs_sobel_x = np.sqrt(sobel_x**2)
        abs_sobel_y = np.sqrt(sobel_y**2)
        # calculate direction with arctan2
        gradient_direction = np.arctan2(abs_sobel_y, abs_sobel_x)
        # create binary mask
        binary_mask = np.zeros_like(gradient_direction)
        # apply threshold
        binary_mask = self.__apply_threshold(binary_mask, gradient_direction, threshold)
        # return binary mask
        return binary_mask
