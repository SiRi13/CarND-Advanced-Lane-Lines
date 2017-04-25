import os
import cv2

import advanced_lane_finder.constants as consts

class AbstractBaseClass():

    def __init__(self, debug_folder,  settings,  verbose,  export):
        self.debug_output_folder = debug_folder
        self.settings = settings
        self.export_images = export
        self.verbose = verbose

    def _debug_msg(self, msg):
        if self.verbose:
            print(msg)

    def _save_image(self, image, folder, idx):
        if self.export_images:
            output = os.path.join(self.debug_output_folder, consts.TEST_IMG_EXPORT_NAME.format(folder, idx))
            os.makedirs(os.path.dirname(output), exist_ok=True)
            if self.verbose:
                print("Export to: ", output)
            mpimg.imsave(output, image)

    def get_settings(self):
        return self.settings
