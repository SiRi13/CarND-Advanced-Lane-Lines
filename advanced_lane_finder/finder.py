import os
import glob
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

from advanced_lane_finder.abstract import AbstractBaseClass
from advanced_lane_finder.frame import Frame
import advanced_lane_finder.constants as consts

class AdvancedLaneFinder(AbstractBaseClass):

    def __init__(self, output_folder, *args, **kwargs):
        AbstractBaseClass.__init__(self, *args, **kwargs)
        self.output_folder = output_folder

    def process_video(self, video_file):
        frame = Frame(self.debug_output_folder, self.settings, self.verbose, self.export)
        if os.path.exists(video_file):
            output = os.path.join(self.output_folder , os.path.basename(video_file))
            os.makedirs(os.path.dirname(output), exist_ok=True)
            self._debug_msg('save to: {}'.format(output))
            videoClip = VideoFileClip(video_file)
            output_clip = videoClip.fl_image(lambda frm: frame.process_frame(frm))
            output_clip.write_videofile(output, audio=False)

    def test_images(self, input_folder):
        frame = Frame(self.debug_output_folder, self.settings, self.verbose, self.export)
        for test_img_path in glob.glob(os.path.join(input_folder, '*.jpg')):
            test_img = mpimg.imread(test_img_path)
            result = frame.process_frame(test_img, True)
            filename = os.path.basename(test_img_path).replace('jpg', 'png')
            filename = os.path.join(self.output_folder, filename)
            self._debug_msg("save image: {}".format(filename))
            mpimg.imsave(filename, result, format='png')
