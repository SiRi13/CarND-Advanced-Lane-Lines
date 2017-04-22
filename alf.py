import argparse
import advanced_lane_finder.alf_constants as consts
import advanced_lane_finder.alf_calibration as calibration

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
