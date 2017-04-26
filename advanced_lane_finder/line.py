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
        self.history = np.zeros((3, consts.HISTORY_SIZE), dtype=np.float32)
        self.poly_fit = np.zeros(3, dtype=np.float32)

        self.radius_m = None
        self.off_center_m = None

        self.pixel_per_meter = self.settings[consts.KEY_PIXEL_PER_METER]
        self.meter_per_pixel = np.power(self.pixel_per_meter, -1)

        self.line_off_center = line_offset
        self.warped_size = warped_size
        self.n_windows = consts.NUMBER_OF_WINDOWS
        self.window_width = consts.WINDOW_WIDTH
        self.window_height = self.warped_size[1] // self.n_windows

        self.line_mask = np.ones((self.warped_size[1], self.warped_size[0]), dtype=np.uint8)
        self.mirrored_line_mask = np.zeros_like(self.line_mask)
        self.line = np.zeros_like(self.line_mask)

    def __negative(self):
        self.found_thresh = consts.FOUND_THRESHOLD
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
        return (self.warped_size[0]//2) / self.pixel_per_meter[0] + self.line_off_center

    def search(self, binary, reset=False):
        # do full search
        if reset or (not self.found and self.found_thresh == consts.FOUND_THRESHOLD) or self.first:
            self.line_mask[:] = 0
            win_y = self.warped_size[1] - 4 * self.window_height
            win_height = self.warped_size[1] - 5
            win_x = self.warped_size[0] // 2 + self.line_off_center * self.pixel_per_meter[0] - 3 * self.window_width
            win_width = win_x + 6 * self.window_width
            conv_a = np.sum(binary[int(win_y):int(win_height), int(win_x):int(win_width)], axis=0)
            conv_v = np.ones(self.window_width) / self.window_width
            conv_signal = np.convolve(conv_a, conv_v, mode='same')
            conv_center = win_x + np.argmax(conv_signal)
            shift = 0
            for y_position in range(self.warped_size[1], 0, -self.window_height):
                win_y = max(0, y_position - 4 * self.window_height)
                conv_a = np.sum(binary[win_y:y_position, :], axis=0)
                conv_signal = np.convolve(conv_a, conv_v, mode='same')
                min_idx = min(max(conv_center + int(shift) - self.window_width//2, 0), self.warped_size[0]-1)
                max_idx = min(max(conv_center + int(shift) + self.window_width//2, 1), self.warped_size[0])
                new_conv_center = min_idx + np.argmax(conv_signal[int(min_idx):int(max_idx)])
                new_max = np.max(conv_signal[int(min_idx):int(max_idx)])
                if new_max <= 2:
                    new_conv_center = conv_center + int(shift)
                    shift = shift/2
                if y_position != self.warped_size[1]:
                    shift = shift / 4 + 0.75*(new_conv_center - conv_center)
                conv_center = new_conv_center
                p1 = int(conv_center - self.window_width//2), int(y_position - self.window_height)
                p2 = int(conv_center + self.window_width//2), int(y_position)
                cv2.rectangle(self.line_mask, p1, p2, 1, thickness=-1)
        else:
            self.line_mask[:] = 0
            th = int(2 * self.window_width) if self.found else int(3 * self.window_width)
            cv2.polylines(self.line_mask, [self.get_points()], 0, 1, thickness=th)

        # save line mask
        title = 'left' if self.line_off_center < 0 else 'right'
        self._save_image(self.line_mask, 'line_mask_' + title, self.count, color_map='gray')

        self.line = self.line_mask * binary
        self._save_image(self.line, 'line_' + title, self.count, color_map='gray')

        self.fit(self.line)
        self.first = False

        if not self.found:
            self.line_mask[:] = 1

        points = self.get_mirrored_points()
        self.mirrored_line_mask[:] = 0
        cv2.polylines(self.mirrored_line_mask, [points], 0, 1, thickness=int(5*self.window_width))
        self._save_image(self.mirrored_line_mask, 'mirrored_line_mask_' + title, self.count, color_map='gray')

    def fit(self, binary):
        y, x = np.where(binary)
        y = y.astype(np.float32)/self.pixel_per_meter[1]
        x = x.astype(np.float32)/self.pixel_per_meter[0]
        if len(y) <= consts.MIN_Y_COUNT:
            coeffs = np.array([0, 0, self.__get_poly_bias()], dtype=np.float32)
        else:
            coeffs, v = np.polyfit(y, x, 2, rcond=1e-16, cov=True)
            self.deviation = 1 - math.exp(-5*np.sqrt(np.trace(v)))

        self.history = np.roll(self.history, 1)
        if self.first:
            self.history = np.reshape(np.repeat(coeffs, consts.HISTORY_SIZE), (3, consts.HISTORY_SIZE))
        else:
            self.history[:, 0] = coeffs

        x_center = self.calculate_position_m(coeffs)
        radius_m = self.calculate_radius_m(coeffs)

        self._debug_msg("position: {}".format(x_center - self.line_off_center))
        abs_center = math.fabs(x_center - self.line_off_center)
        abs_max_off = math.fabs(0.5 * self.line_off_center)
        if (self.deviation > consts.MAX_DEVIATION) | (len(y) < consts.MIN_Y_COUNT) | (radius_m < consts.MIN_RADIUS_M) | (abs_center > abs_max_off):
            self._debug_msg("{}, {}, {}, {}".format(self.deviation, len(y), (abs_center-abs_max_off), radius_m))
            self.history[0:2, 0] = 0
            self.history[2, 0] = self.__get_poly_bias()
            self.__negative()
        else:
            self.__positive()

        self.poly_fit = np.mean(self.history, axis=1)

    def reset(self):
        self._debug_msg('reset line: {}'.format(self.line_off_center))
        self.found = False
        self.poly_fit = np.zeros(3, dtype=np.float32)
        self.line_mask[:] = 1
        self.first = True

    def get_points(self):
        y = np.array(range(0, self.warped_size[1] + 1, 10), dtype=np.float32) / self.pixel_per_meter[1]
        x = np.polyval(self.poly_fit, y) * self.pixel_per_meter[0]
        y *= self.pixel_per_meter[1]
        return np.array([x, y], dtype=np.int32).T

    def get_mirrored_points(self):
        pts = self.get_points()
        pts[:, 0] = pts[:, 0] - 2 * self.line_off_center * self.pixel_per_meter[0]
        return pts

    def use_other_line(self, other_line):
        new_points = other_line.get_mirrored_points()
        x, y = np.where(new_points)
        y = y.astype(np.float32) / self.pixel_per_meter[1]
        x = x.astype(np.float32) / self.pixel_per_meter[0]
        new_poly, v = np.polyfit(y, x, 2, rcond=1e-16, cov=True)
        self.deviation = 1 - math.exp(-5*np.sqrt(np.trace(v)))
        self._debug_msg('other line poly: {}'.format())
        self.history = np.roll(self.history, 1)
        self.history[:, 0] = new_poly
        self.__positive()
        self.poly_fit = np.mean(self.history, axis=1)
        self._debug_msg('new mean poly: {:>6.2f}'.format(self.poly_fit))

    def calculate_radius_m(self, poly):
        warped_y_m = self.warped_size[1] / self.pixel_per_meter[1]
        self.radius_m = ((1 + (2 * poly[0] * warped_y_m + poly[1])**2)**1.5) / np.absolute(2*poly[0])
        return self.radius_m

    def calculate_position_m(self, poly):
        warped_y_m = self.warped_size[1] / self.pixel_per_meter[1]
        self.off_center_m = np.polyval(poly, warped_y_m) - (self.warped_size[0]//2) / self.pixel_per_meter[0]
        return self.off_center_m
