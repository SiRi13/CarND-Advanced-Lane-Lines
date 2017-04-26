import pickle
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import advanced_lane_finder.constants as consts
from advanced_lane_finder.calibration import CameraCalibration
from advanced_lane_finder.transformation import ImageTransformation
from advanced_lane_finder.frame import Frame
from advanced_lane_finder.finder import AdvancedLaneFinder

# %% load settings
settings_pickle_path = os.path.join(consts.SETTINGS_FOLDER, consts.SETTINGS_PICKLE)
settings = pickle.load(open(settings_pickle_path, mode='rb'))
debug_folder = './debug_images/'
input_folder = './test_images/'

settings[consts.KEY_TIME_STAMP_CALIBRATION]
settings[consts.KEY_TIME_STAMP_TRANSFORMATION]
# settings[consts.KEY_PIXEL_PER_METER][0] * 1.8288
# np.power(settings[consts.KEY_PIXEL_PER_METER], -1)

# %% test calibration
if False:
    print("init calibration class")
    cam_cal = CameraCalibration(debug_folder, settings, True, False)
    print("start camera calibration")
    cam_cal.calibrate_camera()
    print("save settings")
    cam_cal.save_calibration(True)

# %% transformation
if False:
    print('init transformation')
    it = ImageTransformation(settings, input_folder, True, False)
    print('find matrix')
    it.find_transformation_matrix(True)
    print('determine pixel per meter')
    it.get_pixel_per_meter(True)
    it.save_transformation_settings(True)

# %% test frame
if False:
    test_img = mpimg.imread('./test_images/test3.jpg')
    frm = Frame(debug_folder, settings, True, False)
    result = frm.process_frame(test_img)

    plt.figure(figsize=(20, 15))
    plt.imshow(result)
    plt.show()

if True:
    output_folder = os.path.join(debug_folder, 'test')
    input_folder = os.path.join(input_folder, 'single_image')
    alf = AdvancedLaneFinder(output_folder, debug_folder, settings, True, True)
    alf.test_images(input_folder)
