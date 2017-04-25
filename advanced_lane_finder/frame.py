import cv2
import numpy as np
import matplotlib.pyplot as plt

from advanced_lane_finder.abstract import AbstractBaseClass
import advanced_lane_finder.constants as consts
from advanced_lane_finder.line import Line

class Frame(AbstractBaseClass):

    def __init__(self, *args, **kwargs):
        AbstractBaseClass.__init__(self, *args, **kwargs)

        self.image_size = consts.IMAGE_SIZE
        self.warped_size = consts.WARPED_SIZE

        self.mask = np.zeros((consts.WARPED_SIZE[1], consts.WARPED_SIZE[0], 3), dtype=np.uint8)
        self.roi_mask = np.ones((consts.WARPED_SIZE[1], consts.WARPED_SIZE[0], 3), dtype=np.uint8)
        self.total_mask = np.zeros((consts.WARPED_SIZE[1], consts.WARPED_SIZE[0], 3), dtype=np.uint8)

        self.left_line = Line(self.warped_size, -1.8288, *args, **kwargs)
        self.right_line = Line(self.warped_size, 1.8288, *args, **kwargs)

    def __undistort(self, img):
        return cv2.undistort(img, self.settings[consts.KEY_CALIBRATION_MATRIX], self.settings[consts.KEY_DISTORTION_COEFFICIENT])

    def __warp(self, img, img_size=consts.WARPED_SIZE, flags=cv2.WARP_FILL_OUTLIERS+cv2.INTER_CUBIC):
        return cv2.warpPerspective(img, self.settings[consts.KEY_TRANSFORMATION_MATRIX], img_size, flags=flags)

    def __blur(self, img):
        img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        # plt.imshow(np.hstack((img_hls, img_lab)), cmap='hot')
        return cv2.medianBlur(img_hls, 5), cv2.medianBlur(img_lab, 5)

    def __determine_largest_contour(self, contours):
        biggest_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > biggest_area:
                biggest_area = area
                biggest_contour = contour

        return biggest_contour

    def __apply_morph_Ex(self, img, kernel_shape=cv2.MORPH_RECT, kernel_size=(3, 3), morph_op=cv2.MORPH_TOPHAT):
        kernel = cv2.getStructuringElement(shape=kernel_shape, ksize=kernel_size)
        return cv2.morphologyEx(img, morph_op, kernel)

    def __filter_curb(self, img, hls, lab):
        # get curb
        curb_mask = (np.uint8(lab[:, :, 2]) > 130) & cv2.inRange(hls, (0, 0, 50), (35, 190, 255))
        # get road
        road_mask = np.uint8(np.logical_not(curb_mask)) & (hls[:, :, 1] < 250)
        # apply morphEx OPEN on road with 7x7 kernel
        morphEx = self.__apply_morph_Ex(road_mask, kernel_size=(7, 7), morph_op=cv2.MORPH_OPEN)
        # dilate morphed road with 31x31 kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        dilation = cv2.dilate(morphEx, kernel)
        # get contours
        img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # get biggest contour/area
        max_contour = self.__determine_largest_contour(contours)

        if self.verbose:
            plt.imshow(cv2.drawContours(dilation, contours, -1, (255, 0, 0), thickness=5), cmap='gray')
            plt.show()

        # fill largest contour on road mask
        road_mask = np.zeros_like(road_mask)
        road_mask = cv2.fillPoly(road_mask, [max_contour],  1)
        return road_mask

    def __morphology_transformation(self, filtered_img, hls, lab):
        # merge lines with road
        self.roi_mask[:, :, 0] = (self.left_line.line_mask | self.right_line.line_mask) & filtered_img
        self.roi_mask[:, :, 1] = self.roi_mask[:, :, 0]
        self.roi_mask[:, :, 2] = self.roi_mask[:, :, 0]

        if self.verbose:
            plt.imshow(self.roi_mask[:, :, 0], cmap='gray')
            plt.title('roi_mask[:, :, 2]')
            plt.show()

        # morphEx on lab and hls
        black = self.__apply_morph_Ex(lab[:, :, 0], cv2.MORPH_ELLIPSE, kernel_size=(7, 3))
        lines = self.__apply_morph_Ex(hls[:, :, 1], cv2.MORPH_ELLIPSE, kernel_size=(7, 3))

        if self.verbose:
            plt.imshow(np.hstack((black, lines)), cmap='gray')
            plt.title('np.hstack((black, lines))')
            plt.show()

        # apply morphEx TOP_HAT with 13x13 kernel for yellow lines
        lines_yellow = self.__apply_morph_Ex(lab[:, :, 2], kernel_size=(13, 13))

        if self.verbose:
            plt.imshow(lines_yellow, cmap='gray')
            plt.title('lines_yellow')
            plt.show()

        # threshold lines
        self.mask[:, :, 0] = cv2.adaptiveThreshold(black, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, -6)
        self.mask[:, :, 1] = cv2.adaptiveThreshold(lines, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, -4)
        self.mask[:, :, 2] = cv2.adaptiveThreshold(lines_yellow, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, -1.5)

        # merge masks and erode
        self.mask *= self.roi_mask
        self.total_mask = np.uint8(np.any(self.mask, axis=2))
        self.total_mask = self.__apply_morph_Ex(self.total_mask, kernel_shape=cv2.MORPH_ELLIPSE, morph_op=cv2.MORPH_ERODE)

        return self.total_mask

    def __highlight_lane(self, img, thickness=5):
        left_pts = self.left_line.get_points()
        right_pts = self.right_line.get_points()
        area_pts = np.concatenate((left_pts, np.flipud(right_pts)), axis=0)
        lane_mask = np.zeros((self.warped_size[1], self.warped_size[0], 3), dtype=np.uint8)
        if self.left_line.found and self.right_line.found:
            cv2.polylines(lane_mask, [left_pts.astype(np.int32)], False, (255, 0, 0), thickness)
            cv2.polylines(lane_mask, [right_pts.astype(np.int32)], False, (0, 0, 255), thickness)
            cv2.fillPoly(lane_mask, [area_pts.astype(np.int32)], (0, 255, 0))
        return lane_mask

    def __put_text(self, img):
        if self.left_line.found and self.right_line.found:
            mean_poly = (self.left_line.poly_fit + self.right_line.poly_fit) / 2.0
            radius_m = self.left_line.calculate_radius_m(mean_poly)
            off_center_m = self.left_line.get_position_m(mean_poly)

            radius = 'Radius (m): {:>6.2f}'.format(radius_m)
            off_ctr = 'Position (m): {:>6.2f}'.format(off_center_m)

            cv2.putText(img, radius, (100, 50), cv2.FONT_HERSHEY_PLAIN, fontScale=2.5, thickness=5, color=(0, 0, 0))
            cv2.putText(img, radius, (100, 50), cv2.FONT_HERSHEY_PLAIN, fontScale=2.5, thickness=4, color=(255, 255, 255))
            cv2.putText(img, off_ctr, (100, 100), cv2.FONT_HERSHEY_PLAIN, fontScale=2.5, thickness=5, color=(0, 0, 0))
            cv2.putText(img, off_ctr, (100, 100), cv2.FONT_HERSHEY_PLAIN, fontScale=2.5, thickness=4, color=(255, 255, 255))
        else:
            cv2.putText(img, "Lane lost!", (550, 170), cv2.FONT_HERSHEY_PLAIN, fontScale=2.5, thickness=5, color=(0, 0, 0))
            cv2.putText(img, "Lane lost!", (550, 170), cv2.FONT_HERSHEY_PLAIN, fontScale=2.5, thickness=4, color=(255, 255, 255))

        return img

    def process_frame(self, frame, reset=False):
        if reset:
            self._debug_msg("process_frame.reset")
            self.left_line.reset()
            self.right_line.reset()

        # undistort, unwarp, change space, filter
        img = self.__undistort(frame)
        warped = self.__warp(img)
        hls, lab = self.__blur(warped)
        filtered_img = self.__filter_curb(warped, hls, lab)

        if self.verbose and False:
            print("filtered.shape", filtered_img.shape)
            plt.imshow(filtered_img, cmap='gray')
            plt.title('filtered_img')
            plt.show()

        morphed = self.__morphology_transformation(filtered_img, hls, lab)

        if self.verbose:
            print("morphed.shape ", morphed.shape)
            plt.imshow(morphed, cmap='gray')
            plt.title('morphed')
            plt.show()

        morphed_l = np.copy(morphed)
        morphed_r = np.copy(morphed)

        if self.right_line.found:
            morphed_l = morphed_l & np.logical_not(self.right_line.line_mask) & self.right_line.mirrored_line_mask
        if self.left_line.found:
            morphed_r = morphed_r & np.logical_not(self.left_line.line_mask) & self.left_line.mirrored_line_mask

        # search w/ convolutional windows
        self.left_line.find_lane_line(morphed_l, reset)
        self.right_line.find_lane_line(morphed_r, reset)

        highlighted = self.__highlight_lane(img)

        if self.verbose:
            print("highlighted.shape ", highlighted.shape)
            plt.imshow(highlighted)
            plt.title('highlighted')
            plt.show()

        uwarped = self.__warp(highlighted, self.image_size, flags=cv2.WARP_FILL_OUTLIERS+cv2.INTER_CUBIC+cv2.WARP_INVERSE_MAP)

        if self.verbose:
            print("uwarped.shape ", uwarped.shape)
            plt.imshow(uwarped)
            plt.title('uwarped')
            plt.show()

        with_text = self.__put_text(img)
        result = cv2.addWeighted(img, 1, uwarped, 0.5, 0)

        if self.verbose:
            print("result.shape ", result.shape)
            plt.imshow(with_text)
            plt.title('highlighted with text')
            plt.show()

        return result
