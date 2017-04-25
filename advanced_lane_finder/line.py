import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

from advanced_lane_finder.abstract import AbstractBaseClass
import advanced_lane_finder.constants as consts

class Line(AbstractBaseClass):

    def __init__(self, warped_size, line_offset, *args, **kwargs):
        AbstractBaseClass.__init__(self, *args, **kwargs)
        self.found = False
        self.first = True
        self.count = 0
        self.n_lost = 0
        self.found_thresh = 1
        self.deviation = 0
        self.history = np.zeros((3, 7), dtype=np.float32)
        self.poly_fit = np.zeros(3, dtype=np.float32)

        self.radius = None
        self.off_center_m = None

        print(self.settings)
        self.pixel_per_meter = self.settings[consts.KEY_PIXEL_PER_METER]
        self.meter_per_pixel = np.power(self.pixel_per_meter, -1)
        self.line_off_center = line_offset
        # self.image_size = img_size
        self.warped_size = warped_size
        self.n_windows = 16
        self.window_height = self.warped_size[1] // self.n_windows
        self.window_width = 30

        self.line_mask = np.ones((self.warped_size[1], self.warped_size[0]), dtype=np.uint8)
        self.mirrored_line_mask = np.zeros_like(self.line_mask)
        self.line = np.zeros_like(self.line_mask)

    def __window_mask(self, warped, center, level):
        retVal = np.zeros_like(warped)

        y_start = int(warped.shape[0] - (level + 1) * self.window_height)
        y_height = int(warped.shape[0] - (level * self.window_height))
        x_start = max(0, int(center - self.window_width / 2))
        x_width = min(int(center + self.window_width / 2), warped.shape[1])

        retVal[y_start:y_height, x_start:x_width] = 1
        return retVal

    def __find_window_centroids(self, image):
        # centroids positions per level (left, right)
        window_centroids = list()
        # window template for convolutions
        conv_a = np.ones(self.window_width)

        # 1. find starting positions for left/right lanes
        if self.self.line_off_center < 0:
            conv_v = np.sum(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
            center = np.argmax(np.convolve(conv_a, conv_v)) - self.window_width / 2
        if self.self.line_off_center > 0:
            conv_v = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
            center = np.argmax(np.convolve(conv_a, conv_v)) - self.window_width / 2 + int(image.shape[1] / 2)

        # add as first layer
        window_centroids.append(center)

        # iterate over layers looking for max pixel locations
        for level in range(1, self.n_windows):
            # convolve window into vertical slice of image
            conv_v = np.sum(image[int(image.shape[0] - (level + 1) * self.window_height):int(image.shape[0] - (level * self.window_height)), :], axis=0)
            conv_signal = np.convolve(conv_a, conv_v)
            # find best left centroid by using past centroids as a reference
            # conv_signal is the right side of window => self.window_width/2
            offset = self.window_width / 2
            if self.self.line_off_center < 0:
                l_min_idx = int(max(center + offset - self.margin, 0))
                l_max_idx = int(min(center + offset + self.margin, image.shape[1]))
                center = np.argmax(conv_signal[l_min_idx:l_max_idx]) + l_min_idx - offset
            if self.self.line_off_center > 0:
                # find best right centroid with past centroid as reference
                r_min_idx = int(max(center + offset - self.margin, 0))
                r_max_idx = int(min(center + offset + self.margin, image.shape[1]))
                center = np.argmax(conv_signal[r_min_idx:r_max_idx]) + r_min_idx - offset
            # add to list
            window_centroids.append(center)

        return window_centroids

    def convolve_search_old(self, warped):
        window_centroids = self.__find_window_centroids(warped)

        self.points = np.zeros_like(warped)

        # iterate levels and draw windows
        for level in range(len(window_centroids)):
            # draw window areas
            line_mask = self.__window_mask(warped, window_centroids[level], level)
            # add graphic points to masks
            self.points[(self.points == 255) | (line_mask == 1)] = 255

        return self.points

    def skip_search_old(self, warped):
        non_zero = warped.nonzero()
        non_zero_y = np.array(non_zero[0])
        non_zero_x = np.array(non_zero[1])

        if self.self.line_off_center < 0:
            lane_indcs = ((non_zero_x > (self.fit[0] * (non_zero_y**2) + self.fit[1] * non_zero_y + self.fit[2] - self.margin))
                            & (non_zero_x < (self.fit[0] * (non_zero_y**2) + self.fit[1] * non_zero_y + self.fit[2] + self.margin)))
        if self.self.line_off_center > 0:
            lane_indcs = ((non_zero_x > (self.fit[0] * (non_zero_y**2) + self.fit[1] * non_zero_y + self.fit[2] - self.margin))
                                & (non_zero_x < (self.it[0] * (non_zero_y**2) + self.fit[1] * non_zero_y + self.fit[2] + self.margin)))

        if not np.any(lane_indcs):
            self.found = False
            return

        self.X, self.Y = non_zero_x[lane_indcs], non_zero_y[lane_indcs]

        self.fit = np.polyfit(self.Y, self.X, 2)

        self.plot_y = np.linspace(0, warped.shape[0] - 1, warped.shape[0])

        return self.fit, self.plot_y

    def fit_line_old(self, warped):
        self.count += 1

        self.Y, self.X = np.where(self.points)

        fit = np.polyfit(self.Y, self.X, 2)
        self.deviation = 1 - math.exp(-5 * np.sqrt(np.trace(cov)))

        self.plot_y = np.linspace(0, warped.shape[0] - 1, warped.shape[0])

        self.history = np.roll(self.history, axis=1)
        if self.first:
            self.history = np.reshape(np.repeat(fit, 7), (3, 7))
        else:
            self.history[:, 0] = fit

            self.poly_fit = np.mean(self.history, axis=1)
            return self.poly_fit, self.plot_y

    def convolve_search(self, morphed, reset=False):
        if reset or (not self.found and self.found_thresh == 5) or self.first:
            self._debug_msg('convolutional search')

            self.line_mask[:] = 1
            y_start = self.warped_size[1] - self.window_height
            x_start = self.warped_size[0] // 2 + self.line_off_center - 3 * self.window_width
            x_width = x_start + 6 * self.window_width
            self._debug_msg('conv_a: {}, {}, {}'.format(y_start, x_start, x_width))
            conv_a = np.sum(morphed[y_start:, x_start:x_width], axis=0)
            conv_v = np.ones(self.window_width) / self.window_width
            conv_signal = np.convolve(conv_a, conv_a, mode='same')
            conv_center = np.argmax(conv_signal) + x_start
            margin = 0

            for level in range(1, self.n_windows):
                y_start = self.warped_size[0] - (level + 1) * self.window_height
                y_height = self.warped_size[0] - (level * self.window_height)
                self._debug_msg('level" {} conv_a: {}, {}'.format(level, y_start, y_height))
                conv_a = np.sum(morphed[y_start:y_height, :], axis=0)
                conv_signal = np.convolve(conv_a, conv_v, mode='same')
                min_idx = min(max(conv_center + int(margin) - self.window_width // 2, 0), self.warped_size[0] - 1)
                max_idx = min(max(conv_center + int(margin) + self.window_width // 2, 1), self.warped_size[0])
                self._debug_msg('signal min {} / max {}'.format(min_idx, max_idx))
                new_center = y_start + np.argmax(conv_signal[min_idx:max_idx])
                center_max = np.max(conv_signal[min_idx:max_idx])
                if center_max <=2:
                    new_center = conv_center + int(margin)
                    margin = margin / 2
                if level < self.n_windows:
                    margin = margin / 4 + 0.75 * (new_center - conv_center)
                conv_center = new_center

                p1 = (conv_center - self.window_width//2, max(0, self.warped_size[1] - level * self.window_height))
                p2 = (conv_center + self.window_width//2, min(self.warped_size[1], self.warped_size[1] + (level+1) * self.window_height))
                self._debug_msg('window p1 {}\tp2 {}'.format(p1, p2))
                cv2.rectangle(self.line_mask, p1, p2, 1, thickness=3)
        else:
            self._debug_msg('skipped search')

            self.line_mask[:] = 0
            points = self.get_points()
            if not self.found:
                th = int(3 * self.window_width)
            else:
                th = int(2 * self.window_width)

            cv2.polylines(self.line_mask, [points], 0, 1, thickness=th)

        self._debug_msg('sarting to fit line')
        self.line = self.line_mask * morphed
        self.fit_line()
        self.first = False
        if not self.found:
            self._debug_msg('line not found!')
            self.line_mask[:] = 1
        points = self.get_mirrored_points()
        self.mirrored_line_mask[:] = 0
        cv2.polylines(self.mirrored_line_mask, [points], 0, 1, thickness=int(5*self.window_width))

    def find_lane_line(self, mask, reset=False):
        step = self.warped_size[1]//self.n_windows

        if reset or (not self.found and self.found_thresh == 5) or self.first:
            self.line_mask[:] = 0
            n_steps = 4
            window_start = self.warped_size[0]//2 + int(self.line_off_center*self.pixel_per_meter[0]) - 3 * self.window_width
            window_end = window_start + 6*self.window_width
            sm = np.sum(mask[self.warped_size[1]-4*step:self.warped_size[1], window_start:window_end], axis=0)
            sm = np.convolve(sm, np.ones((self.window_width,))/self.window_width, mode='same')
            argmax = window_start + np.argmax(sm)
            shift = 0
            for last in range(self.warped_size[1], 0, -step):
                first_line = max(0, last - n_steps*step)
                sm = np.sum(mask[first_line:last, :], axis=0)
                sm = np.convolve(sm, np.ones((self.window_width,))/self.window_width, mode='same')
                window_start = min(max(argmax + int(shift)-self.window_width//2, 0), self.warped_size[0]-1)
                window_end =   min(max(argmax + int(shift) + self.window_width//2, 0+1), self.warped_size[0])
                new_argmax = window_start + np.argmax(sm[window_start:window_end])
                new_max = np.max(sm[window_start:window_end])
                if new_max <= 2:
                    new_argmax = argmax + int(shift)
                    shift = shift/2
                if last != self.warped_size[1]:
                    shift = shift*0.25 + 0.75*(new_argmax - argmax)
                argmax = new_argmax
                cv2.rectangle(self.line_mask, (argmax-self.window_width//2, last-step), (argmax+self.window_width//2, last),
                              1, thickness=-1)
        else:
            self.line_mask[:] = 0
            points = self.get_points()
            if not self.found:
                factor = 3
            else:
                factor = 2
            cv2.polylines(self.line_mask, [points], 0, 1, thickness=int(factor*self.window_width))

        self.line = self.line_mask * mask
        self.fit_lane_line(self.line)
        self.first = False
        if not self.found:
            self.line_mask[:] = 1
        points = self.get_mirrored_points()
        self.mirrored_line_mask[:] = 0
        cv2.polylines(self.mirrored_line_mask, [points], 0, 1, thickness=int(5*self.window_width))

    def fit_lane_line(self, mask):
        y_coord, x_coord = np.where(mask)
        y_coord = y_coord.astype(np.float32)/self.pixel_per_meter[1]
        x_coord = x_coord.astype(np.float32)/self.pixel_per_meter[0]
        if len(y_coord) <= 150:
            coeffs = np.array([0, 0, (self.warped_size[0]//2)/self.pixel_per_meter[0] + self.line_off_center], dtype=np.float32)
        else:
            coeffs, v = np.polyfit(y_coord, x_coord, 2, rcond=1e-16, cov=True)
            self.deviation = 1 - math.exp(-5*np.sqrt(np.trace(v)))

        self.history = np.roll(self.history, 1)

        if self.first:
            self.history = np.reshape(np.repeat(coeffs, 7), (3, 7))
        else:
            self.history[:, 0] = coeffs

        value_x = self.get_center_shift(coeffs, self.warped_size, self.pixel_per_meter)
        curve = self.get_curvature(coeffs, self.warped_size, self.pixel_per_meter)

        self._debug_msg("position: {}".format(value_x - self.line_off_center))
        if (self.deviation > 0.95) | (len(y_coord) < 150) | (math.fabs(value_x - self.line_off_center) > math.fabs(0.5*self.line_off_center)) \
                | (curve < 30):

            self.history[0:2, 0] = 0
            self.history[2, 0] = (self.warped_size[0]//2)/self.pixel_per_meter[0] + self.line_off_center
            self.__negative()
            self._debug_msg("{}, {}, {}, {}".format(self.deviation, len(y_coord), math.fabs(value_x-self.line_off_center)-math.fabs(0.5*self.line_off_center), curve))
        else:
            self.__positive()

        self.poly_fit = np.mean(self.history, axis=1)

    def fit_line(self):
        self.Y, self.X = np.where(self.line)

        if len(self.Y) <= 150:
            self.poly_fit = np.array([0, 0, self.__get_poly_bias()], dtype=np.float32)
        else:
            self.poly_fit = np.polyfit(self.Y, self.Y, 2)

        self.history = np.roll(self.history, 1)
        if self.first:
            self.history = np.reshape(np.repeat(self.poly_fit, 7), (3, 7))
        else:
            self.history[:, 0] = self.poly_fit

        off_center_m = self.get_position_m()
        radius = self.calculate_radius_m()
        self._debug_msg('off_ctr - line_offset: {}'.format(off_center_m - self.line_off_center / self.pixel_per_meter[0]))
        if (len(self.Y) < 150) or (math.fabs(off_center_m - self.line_off_center) > math.fabs(self.line_off_center / 2)) \
            or (radius < 30):
            self.history[0:2, 0] = 0
            self.history[2, 0] = self.__get_poly_bias()
            self.__negative()
            self._debug_msg("line was not found! {}, {}, {}, {}".format(len(self.Y), radius, \
                            math.fabs(off_center_m - self.line_off_center), math.fabs(self.line_off_center / 2)))
        else:
            self._debug_msg('line found!')
            self.__positive()

        self.poly_fit = np.mean(self.history, axis=1)

    def reset(self):
        self._debug_msg('reset line: {}'.format(self.line_off_center))
        self.found = False
        self.poly_fit = np.zeros(3, dtype=np.float32)
        self.line_mask[:] = 1
        self.first = True

    def __negative(self):
        self.found_thresh = 5
        if self.found:
            self.n_lost += 1
            if self.n_lost > 6:
                self.reset()

    def __positive(self):
        self.first = False
        self.n_lost = 0
        if not self.found:
            self.found_thresh -= 1
            self.found = True if self.found_thresh <= 0 else False

    def __get_poly_bias(self):
        return (self.warped_size[0] // 2) + self.line_off_center

    def get_points(self):
        y = np.array(range(0, self.warped_size[1] + 1, 10), dtype=np.float32) / self.pixel_per_meter[1]
        x = np.polyval(self.poly_fit, y) * self.pixel_per_meter[0]
        y *= self.pixel_per_meter[1]
        return np.array([x, y], dtype=np.int32).T

    def get_mirrored_points(self):
        pts = self.get_points()
        pts[:, 0] = pts[:, 0] - 2 * self.line_off_center * self.pixel_per_meter[0]
        return pts

    def calculate_radius_m(self, mean_poly=None):
        mean_poly = self.poly_fit if mean_poly is None else mean_poly
        if self.count % 3 == 0:
            Y, X = np.where(self.line)
            fit_meter = np.polyfit(Y / self.pixel_per_meter[1], X / self.pixel_per_meter[1], 2)
            warped_y_m = self.warped_size[1] / self.pixel_per_meter[1]
            self.radius_m = ((1 + (2 * fit_meter[0] * warped_y_m + fit_meter[1])**2)**1.5) / np.absolute(2 * fit_meter[0])

        return self.radius_m

    def get_position_m(self, mean_poly=None):
        mean_poly = self.poly_fit if mean_poly is None else mean_poly
        if self.count % 3 == 0:
            warped_y_m = self.warped_size[1] / self.pixel_per_meter[1]
            self.off_center_m = np.polyval(mean_poly, warped_y_m) - (self.warped_size[0] // 2) / self.pixel_per_meter[0]
        return self.off_center_m

    def get_center_shift(self, coeffs, img_size, pixels_per_meter):
        return np.polyval(coeffs, img_size[1]/pixels_per_meter[1]) - (img_size[0]//2)/pixels_per_meter[0]

    def get_curvature(self, coeffs, img_size, pixels_per_meter):
        return ((1 + (2*coeffs[0]*img_size[1]/pixels_per_meter[1] + coeffs[1])**2)**1.5) / np.absolute(2*coeffs[0])
