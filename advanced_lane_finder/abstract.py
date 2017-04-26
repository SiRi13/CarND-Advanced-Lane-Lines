import os
import cv2
import matplotlib.image as mpimg

import advanced_lane_finder.constants as consts

class AbstractBaseClass():

    def __init__(self, debug_folder,  settings,  verbose,  export):
        self.debug_output_folder = debug_folder
        self.settings = settings
        self.verbose = verbose
        self.export = export

    def _debug_msg(self, msg):
        if self.verbose:
            print(msg)

    def _save_image(self, image, source, idx, color_map=None):
        if self.export:
            output = os.path.join(self.debug_output_folder, consts.TEST_IMG_EXPORT_NAME.format(source, idx))
            output.replace('jpg', 'png')
            os.makedirs(os.path.dirname(output), exist_ok=True)
            if self.verbose:
                print("Export to: ", output)
            mpimg.imsave(output, image, format='png', cmap=color_map)

    def get_settings(self):
        return self.settings
