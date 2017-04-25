import cv2

"""
image settings
"""
IMAGE_SIZE = 1280, 720
WARPED_SIZE = 500, 600

"""
path & filename settings
"""
SETTINGS_FOLDER = './advanced_lane_finder/settings/'
SETTINGS_PICKLE = 'settings.p'
CALIBRATION_IMAGE_TEMPLATE = 'calibration*.jpg'
CALIBRATION_IMAGE_FOLDER = './camera_cal/'
CAL_IMG_EXPORT_NAME='./output_images/calibration_image_{}_{}.png'
DEFAULT_OUTPUT_IMAGE_FOLDER = './output_images/'
DEFAULT_INPUT_IMAGE_FOLDER = './test_images/'
DEFAULT_DEBUG_FOLDER = './debug_images/'
PROJECT_VIDEO = './project_video.mp4'
CHALLENGE_VIDEO = './challenge_video.mp4'
HARDER_CHALLENGE_VIDEO = './harder_challenge_video.mp4'

"""
calibration & transformation settings
"""
DEFAULT_OBJECT_POINT = 9, 6
DEFAULT_SUB_PIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
DEFAULT_SUB_PIX_WIN_SIZE = 11, 11
DEFAULT_SUB_PIX_ZERO_ZONE = -1, 1
DEFAULT_LANE_LINES_DISTANCE_PX = 1000
LANE_WIDTH_M = 3.7

"""
keys for settings pickle
"""
KEY_TIME_STAMP_CALIBRATION = 'time_stamp_calibration'
KEY_TIME_STAMP_TRANSFORMATION = 'time_stamp_transformation'
KEY_OBJECT_POINTS = 'opject_points'
KEY_IMAGE_POINTS = 'image_points'
KEY_IMAGE_SIZE = 'image_size'
KEY_DISTORTION_COEFFICIENT = 'distortion_coefficient'
KEY_CALIBRATION_MATRIX = 'calibration_matrix'
KEY_TRANSFORMATION_MATRIX = 'transformation_matrix'
KEY_TRANSFORMATION_MATRIX_INV = 'transformation_matrix_inv'
KEY_PIXEL_PER_METER = 'pixel_per_meter'
KEY_TRANSFORMATION_SOURCE_POINTS = 'transformation_src_points'
KEY_TRANSFORMATION_DESTINATION_POINTS = 'transformation_dst_points'

KEY_LIST = [KEY_OBJECT_POINTS, KEY_IMAGE_POINTS, KEY_IMAGE_SIZE, KEY_DISTORTION_COEFFICIENT, KEY_CALIBRATION_MATRIX]
