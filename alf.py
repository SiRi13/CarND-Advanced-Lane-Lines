import os
import pickle
import argparse
import advanced_lane_finder.alf_constants as consts
from advanced_lane_finder.alf_calibration import CameraCalibration
from advanced_lane_finder.alf_transformation import ImageTransformation




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ALF - Advanced Lane Finder')
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--init-settings',
        action='store_true',
        help='init settings pickle. old file gets erased!'
    )
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

    settings_pickle_path = os.path.join(consts.SETTINGS_FOLDER, consts.SETTINGS_PICKLE)
    if args.init_settings:
        os.makedirs(os.path.dirname(settings_pickle_path), exist_ok=True)
        settings_pickle = { 'empty': False }
        pickle.dump(settings_pickle, open(settings_pickle_path, mode='wb'))

    # load settings
    assert os.path.exists(settings_pickle_path), 'No settings found! Exit'
    settings_pickle = pickle.load(open(settings_pickle_path, mode='rb'))

    if os.path.exists(args.input_folder):
        input_folder = args.input_folder
    else:
        print('Provided input folder not found! using default')
        input_folder = consts.DEFAULT_INPUT_IMAGE_FOLDER

    if os.path.exists(args.output_folder):
        output_folder = args.output_folder
    else:
        print('Provided output folder not found! using default')
        output_folder = consts.DEFAULT_OUTPUT_IMAGE_FOLDER

    if args.calibrate:
        print("init calibration class")
        cam_cal = CameraCalibration(export=args.verbosity)
        print("start camera calibration")
        cam_cal.calibrate_camera()
        print("save settings")
        cam_cal.save_calibration(args.force)

    elif args.transformation:
        print('init transformation')
        it = ImageTransformation(settings_pickle, input_folder)
        print('find matrix')
        it.find_transformation_matrix(True)
        print('determine pixel per meter')
        it.get_pixel_per_meter(True)
        it.save_transformation_settings(args.force)

    elif args.project:
        print('process project video')
        alf = AdvancedLaneFinder(settings_pickle, output_folder)
        alf.process_video(consts.PROJECT_VIDEO)

    elif args.challenge:
        print('process challenge video')
        alf = AdvancedLaneFinder(settings_pickle, output_folder)
        alf.process_video(consts.CHALLENGE_VIDEO)

    elif args.harder_challenge:
        print('process harder video')
        alf = AdvancedLaneFinder(settings_pickle, output_folder)
        alf.process_video(consts.HARDER_CHALLENGE_VIDEO)

    elif args.test_images:
        print("process test images")
        alf = AdvancedLaneFinder(settings_pickle, output_folder)
        alf.test_images(input_folder, args.verbosity)
