import os
import cv2
import glob
import pickle
import matplotlib.image as mpimg
import numpy as np

import advanced_lane_finder.alf_constants as consts

class CameraCalibration:

    def __init__(self, import_folder=consts.CALIBRATION_IMAGE_FOLDER,
                 import_files=consts.CALIBRATION_IMAGE_TEMPLATE, export=False):
        self.export_calibrated = export
        self.import_folder = import_folder
        self.import_files = import_files

        # init lists etc.
        self.calibration_images = list()
        self.object_points = list()
        self.image_points = list()

        self.__load_calibration_images()

    def __load_calibration_images(self):
        if os.path.exists(self.import_folder):
            for img_path in glob.glob(os.path.join(self.import_folder, self.import_files)):
                # load image
                img = mpimg.imread(img_path)
                # append to list
                self.calibration_images.append(img)

    def calibrate_camera(self, obj_point=consts.DEFAULT_OBJECT_POINT,
                               criteria=consts.DEFAULT_SUB_PIX_CRITERIA,
                               win_size=consts.DEFAULT_SUB_PIX_WIN_SIZE,
                               zero_zone=consts.DEFAULT_SUB_PIX_ZERO_ZONE):
        # init obj_point array
        obj_pt_template = np.zeros((obj_point[0] * obj_point[1], 3), np.float32)
        # fill with values from (0,0,0) ... (nx-1, ny-1, 0)
        obj_pt_template[:,:2] = np.mgrid[0:obj_point[0], 0:obj_point[1]].T.reshape(-1, 2)
        # iterate through images and find corners
        for idx, img in enumerate(self.calibration_images):
            # convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # try to get the corners
            ret, corners = cv2.findChessboardCorners(img, obj_point, None)
            # if corners found
            if ret == True:
                # for debuggin & writeup
                if self.export_calibrated:
                    cv2.drawChessboardCorners(img, obj_point, corners, ret)
                    mpimg.imsave(consts.CAL_IMG_EXPORT_NAME.format('chessboard_corners',idx), img, format='png')

                # refine corner positions
                corners = cv2.cornerSubPix(gray, corners, win_size, zero_zone, criteria)

                # for debuggin & writeup
                if self.export_calibrated:
                    cv2.drawChessboardCorners(img, obj_point, corners, ret)
                    mpimg.imsave(consts.CAL_IMG_EXPORT_NAME.format('chessboard_corners_refined',idx), img, format='png')

                # append to lists
                self.object_points.append(obj_pt_template)
                self.image_points.append(corners)

            del idx, img, corners

        # calibrate
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.object_points, self.image_points, gray.shape[::-1], None, None)
        self.calibration_matrix = mtx
        self.distortion_coefficient = dist
        self.image_size = gray.shape

        if self.export_calibrated:
            for idx, img in enumerate(self.calibration_images):
                destination = cv2.undistort(img, mtx, dist)
                mpimg.imsave(consts.CAL_IMG_EXPORT_NAME.format('undistorted', idx), destination, format='png')
                del idx, img, destination


    def save_calibration(self, force=False):
        settings_pickle_path = os.path.join(consts.SETTINGS_FOLDER, consts.SETTINGS_PICKLE)
        if os.path.exists(consts.SETTINGS_FOLDER):
            pickle_dict = dict()
            do_save = True
            # only check when file exists
            if os.path.exists(settings_pickle_path):
                # have to open it as 'read-binary' for pickle.load()
                with open(settings_pickle_path, 'rb') as i:
                    pickle_dict = pickle.load(i)
                    for key in pickle_dict.keys():
                        if key in consts.KEY_LIST:
                            do_save = force

            if do_save:
                with open(settings_pickle_path, 'wb') as f:
                    pickle_dict[consts.KEY_OBJECT_POINTS] = self.object_points
                    pickle_dict[consts.KEY_IMAGE_POINTS] = self.image_points
                    pickle_dict[consts.KEY_DISTORTION_COEFFICIENT] = self.distortion_coefficient
                    pickle_dict[consts.KEY_CALIBRATION_MATRIX] = self.calibration_matrix
                    pickle_dict[consts.KEY_IMAGE_SIZE] = self.image_size

                    pickle.dump(pickle_dict, f)