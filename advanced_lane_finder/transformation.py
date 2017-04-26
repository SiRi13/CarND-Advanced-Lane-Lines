import os
import cv2
import glob
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from advanced_lane_finder.abstract import AbstractBaseClass
import advanced_lane_finder.constants as consts

class ImageTransformation(AbstractBaseClass):

    def __init__(self, input_folder, output_folder, *args, **kwargs):
        AbstractBaseClass.__init__(self, *args, **kwargs)
        self.input_folder = input_folder
        self.output_folder = output_folder

        self.cal_M = self.settings[consts.KEY_CALIBRATION_MATRIX]
        self.dist = self.settings[consts.KEY_DISTORTION_COEFFICIENT]

        self.__init_values()
        self.__load_images()

    def __init_values(self):
        self.images = list()
        self.image_paths = list()
        self.path_template = os.path.join(self.output_folder, consts.TEST_IMG_EXPORT_NAME)

    def __load_images(self):
        if os.path.exists(self.input_folder):
            self.image_paths = glob.glob(os.path.join(self.input_folder, "straight*.jpg"))
            for img_path in self.image_paths:
                self.images.append(mpimg.imread(img_path))

    def __point_on_line(self, p1, p2, y):
        return [p1[0] + (p2[0] - p1[0]) / float(p2[1] - p1[1]) * (y - p1[1]), y]

    def __calculate_line_intercept(self):
        roi_points = np.array([[0, consts.IMAGE_SIZE[1] - 50], [consts.IMAGE_SIZE[0], consts.IMAGE_SIZE[1] - 50],
                                [consts.IMAGE_SIZE[0] // 2, consts.IMAGE_SIZE[1] // 2 + 50]], dtype=np.int32)
        roi = np.zeros((consts.IMAGE_SIZE[1], consts.IMAGE_SIZE[0]), dtype=np.uint8)
        # fill roi
        cv2.fillPoly(roi, [roi_points], 1)

        Lhs = np.zeros((2, 2), dtype=np.float32)
        Rhs = np.zeros((2, 1), dtype=np.float32)

        # iterate images
        for idx, img in enumerate(self.images):
            # undistort
            undistorted = cv2.undistort(np.copy(img), self.cal_M, self.dist)
            # convert to HLS space
            hls = cv2.cvtColor(undistorted, cv2.COLOR_RGB2HLS)
            # get edges
            canned = cv2.Canny(hls[:, :, 1], 200, 100)
            # get lines in roi
            hughed = cv2.HoughLinesP(canned * roi, 0.5, np.pi/180, 20, None, 180, 120)
            # iterate lines
            for line in hughed:
                # get points on line
                for x1, y1, x2, y2 in line:
                    norm = np.array([[(y2 - y1) * -1], [x2 - x1]], dtype=np.float32)
                    norm /= np.linalg.norm(norm)
                    point = np.array([[x1], [y1]], dtype=np.float32)
                    outer = np.matmul(norm, norm.T)
                    Lhs += outer
                    Rhs += np.matmul(outer, point)
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)

            if self.verbose:
                plt.imshow(img)
                if self.export:
                    plt.savefig(self.path_template.format('line_intercept', idx), format='png')
                plt.show()


        # calculate point where lane lines intercept
        v_point = np.matmul(np.linalg.inv(Lhs), Rhs)
        return v_point

    def find_transformation_matrix(self):
        v_point = self.__calculate_line_intercept()
        # create points
        top = v_point[1] + 60
        bottom = consts.IMAGE_SIZE[1] - 40
        width = 530
        # source_points
        s1 = [v_point[0] - (width / 2), top]
        s2 = [v_point[0] + (width / 2), top]
        s3 = self.__point_on_line(s2, v_point, bottom)
        s4 = self.__point_on_line(s1, v_point, bottom)
        self.source_points = np.array([s1, s2, s3, s4], dtype=np.float32)
        # destination
        d1 = [0, 0]
        d2 = [consts.WARPED_SIZE[0], 0]
        d3 = [consts.WARPED_SIZE[0], consts.WARPED_SIZE[1]]
        d4 = [0, consts.WARPED_SIZE[1]]
        self.destination_points = np.array([d1, d2, d3, d4], dtype=np.float32)
        # generate perspective transformation matrix
        self.M = cv2.getPerspectiveTransform(self.source_points, self.destination_points)
        # and inverse
        self.Minv = cv2.getPerspectiveTransform(self.destination_points, self.source_points)

        if self.verbose:
            img1 = np.copy(self.images[0])
            img1 = cv2.polylines(img1, [np.int32(self.source_points)], True, (0, 0, 255), thickness=4)
            img1 = cv2.polylines(img1, [np.int32(self.destination_points)], True, (0, 255, 0), thickness=4)
            # img1 = cv2.circle(img1, v_point, 5, (255, 0, 0), thickness=-1)
            img2 = np.copy(self.images[1])
            img2 = cv2.polylines(img2, [np.int32(self.source_points)], True, (0, 0, 255), thickness=4)
            img2 = cv2.polylines(img2, [np.int32(self.destination_points)], True, (0, 255, 0), thickness=4)
            # img2 = cv2.circle(img2, v_point, 5, (255, 0, 0), thickness=-1)
            test_output = np.hstack((img1, img2))
            plt.imshow(test_output)
            if self.export:
                plt.savefig(self.path_template.format('all_lines_imgs', 0), format='png')
            plt.show()

        return self.M, self.Minv

    def get_pixel_per_meter(self):
        if self.verbose:
            debug_imgs = list()

        min_distance = consts.DEFAULT_LANE_LINES_DISTANCE_PX
        for img in self.images:
            undistorted = cv2.undistort(np.copy(img), self.cal_M, self.dist)
            warped = cv2.warpPerspective(undistorted, self.M, consts.WARPED_SIZE)
            hls = cv2.cvtColor(warped, cv2.COLOR_RGB2HLS)
            h_bin = hls[:, :, 1] > 128
            h_bin[:, :50] = 0
            h_bin[:, -50:] = 0
            moms = cv2.moments(np.uint8(h_bin[:, :consts.WARPED_SIZE[0] // 2]))
            x1 = moms['m10'] / moms['m00']
            moms = cv2.moments(np.uint8(h_bin[:, consts.WARPED_SIZE[0] // 2:]))
            x2 = consts.WARPED_SIZE[0] // 2 + moms['m10'] / moms['m00']
            cv2.line(warped, (int(x1), 0), (int(x1), consts.WARPED_SIZE[1]), (255, 0, 0), 3)
            cv2.line(warped, (int(x2), 0), (int(x2), consts.WARPED_SIZE[1]), (0, 255, 0), 3)

            if self.verbose:
                debug_imgs.append(warped)

            if ((x2-x1) < min_distance):
                min_distance = x2 - x1

        px_per_m_x = min_distance / consts.LANE_WIDTH_M
        Lh = np.linalg.inv(np.matmul(self.M, self.cal_M))
        px_per_m_y = px_per_m_x * np.linalg.norm(Lh[:, 0]) / np.linalg.norm(Lh[:, 1])

        self.pixel_per_meter = px_per_m_x, px_per_m_y

        if self.verbose:
            self._debug_msg("Pixel per meter (x/y): {}".format(self.pixel_per_meter))
            plt.imshow(np.hstack(tuple(debug_imgs)))
            if self.export:
                plt.savefig(self.path_template.format('pixel_meter', 0), format='png')
            plt.show()

        return self.pixel_per_meter

    def save_transformation_settings(self, do_save):
        self.settings_path = os.path.join(consts.SETTINGS_FOLDER, consts.SETTINGS_PICKLE)
        if os.path.exists(consts.SETTINGS_FOLDER) and do_save:
            with open(settings_pickle_path, 'wb') as f:
                self.settings[consts.KEY_TIME_STAMP_TRANSFORMATION] = time.ctime()
                self.settings[consts.KEY_TRANSFORMATION_MATRIX] = self.M
                self.settings[consts.KEY_TRANSFORMATION_MATRIX_INV] = self.Minv
                self.settings[consts.KEY_PIXEL_PER_METER] = self.pixel_per_meter
                self.settings[consts.KEY_TRANSFORMATION_SOURCE_POINTS] = self.source_points
                self.settings[consts.KEY_TRANSFORMATION_DESTINATION_POINTS] = self.destination_points
                self._debug_msg('save settings')
                pickle.dump(self.settings, f)
