import cv2

"""
image settings
"""
IMAGE_SIZE=(1280, 720)

"""
path & filename settings
"""
SETTINGS_FOLDER = './advanced_lane_finder/settings/'
SETTINGS_PICKLE = 'settings.p'
CALIBRATION_IMAGE_TEMPLATE = 'calibration*.jpg'
CALIBRATION_IMAGE_FOLDER = './camera_cal/'
CAL_IMG_EXPORT_NAME='./output_images/calibration_image_{}_{}.jpg'

"""
calibration settings
"""
DEFAULT_OBJECT_POINT = (9, 6)
DEFAULT_SUB_PIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
DEFAULT_SUB_PIX_WIN_SIZE = (5, 5)
DEFAULT_SUB_PIX_ZERO_ZONE = (-1, 1)
KEY_OBJECT_POINTS = 'opject_points'
KEY_IMAGE_POINTS = 'image_points'
KEY_IMAGE_SIZE = 'image_size'
KEY_DISTORTION_COEFFICIENT = 'distortion_coefficient'
KEY_CALIBRATION_MATRIX = 'calibration_matrix'
