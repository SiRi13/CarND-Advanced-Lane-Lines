import argparse
import advanced_lane_finder.alf_constants as consts
from advanced_lane_finder.alf_calibration import CameraCalibration
from advanced_lane_finder.alf_binary import BinaryImage

class AdvancedLaneFinder:
    def __init__(self, img_size, dest_size, cal_matrix, dist_coeffs, trans_matrix, px_per_m):
        self.image_size = img_size
        self.destination_size = dest_size
        self.calibration_matrix = cal_matrix
        self.distortion_coefficient = dist_coeffs
        self.transformation_matrix = trans_matrix
        self.pixel_per_meter = px_per_m

        self.__init_values()

    def __init_values():
        self.mask = np.zeros((self.destination_size[1], self.destination_size[0], 3), dtype=np.uint8)
        self.roi_mask = np.zeros_like(self.mask, dtype=np.uint8)
        self.total_mask = np.zeros_like(self.roi_mask, dtype=np.uint8)
        self.destination_mask = np.zeros((self.destination_size[1], self.destination_size[0]), dtype=np.uint8)
        self.count = 0
        self.found = False
        self.default_kernel = 15

    def _undistort(self, img):
        return cv2.undistort(img, self.calibration_matrix, self.distortion_coefficient)

    def _transform(self, img, unwarp=False):
        if not unwarp:
            return cv2.warpPerspective(img, self.transformation_matrix, self.destination_size,
                                        flags=cv2.WARP_FILL_OUTLIERS+cv2.INTER_CUBIC)
        if unwarp:
            return cv2.warpPerspective(img, self.transformation_matrix, self.image_size,
                                        flags=cv2.WARP_FILL_OUTLIERS+cv2.INTER_CUBIC+cv2.WARP_INVERSE_MAP)

    def _gradient_threshold(self, img):
        binary_image = BinaryImage(gray, self.default_kernel)

        return binary_image

    def _threshold(self, img):
        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # convert to hls scale
        hls = cv2.cvtColor(gray, cv2.COLOR_RGB2HLS)
        # extract s_channel
        S_channel = hls[:, :, 2]
        # apply gradien thresholding
        binary_grad_image = self._gradient_threshold(binary_color_image)
        return binary_grad_image

    def process_image(self, img):
        # undistort image
        dest = self._undistort(img)
        # transform image
        transformed = self._transform(dest)
        # threshold image
        binary = self._threshold(warped)
        #

        # transform back
        transformed = self._transform(binary, unwarp=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ALF - Advanced Lane Finder')
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--calibrate', '-c',
        action='store_true',
        help='calibrate camera. pass --f --force flag to overwrite old settings.'
    )
    group.add_argument(
        '--transformation', '-t',
        action='store_true',
        help='create transformation matrix. pass -f --force flag to overwrite old settings.'
    )
    group.add_argument(
        '--test-images', '-i',
        action='store_true',
        help='run ALF on the test images in folder "{}" or provide folder with -if --input-folder.'.format(consts.DEFAULT_INPUT_IMAGE_FOLDER)
    )
    group.add_argument(
        '--project',
        action='store_true',
        help='run ALF on the project video "{}".'.format(consts.PROJECT_VIDEO)
    )
    group.add_argument(
        '--challenge',
        action='store_true',
        help='run ALF on the challenge video "{}".'.format(consts.CHALLENGE_VIDEO)
    )
    group.add_argument(
        '--harder-challenge',
        action='store_true',
        help='run ALF on the harder challenge video "{}".'.format(consts.HARDER_CHALLENGE_VIDEO)
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='force saving of settings for example new calibration settings.'
    )
    parser.add_argument(
        '--verbosity', '-v',
        action='store_true',
        help='activate all outputs'
    )
    parser.add_argument(
        '--output-folder', '-of',
        type=str,
        default=consts.DEFAULT_OUTPUT_IMAGE_FOLDER,
        help='Path to output folder. Default: [{}]'.format(consts.DEFAULT_OUTPUT_IMAGE_FOLDER)
    )
    parser.add_argument(
        '--input-folder', '-if',
        type=str,
        default=consts.DEFAULT_INPUT_IMAGE_FOLDER,
        help='Specify folder to load test images from. Default: [{}]'.format(consts.DEFAULT_INPUT_IMAGE_FOLDER)
    )
    args = parser.parse_args()

    if args.calibrate:
        print("init calibration class")
        cam_cal = calibration.CameraCalibration(export=args.verbosity)
        print("start camera calibration")
        cam_cal.calibrate_camera()
        print("save settings")
        cam_cal.save_calibration(args.force)

    elif args.project:
        print('project')
