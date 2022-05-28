from logging import Logger

from engine import PluginCore
from model import Meta

import numpy as np
import cv2
import dlib

class Brightness(PluginCore):

    def __init__(self, logger: Logger, appCore) -> None:
        super().__init__(logger, appCore)
        self.meta = Meta(
            name='Brightness plugin',
            description='Plugin for Brightness level',
            version='0.0.1'
        )
        self.appCore = appCore
        self.CROP_ALPHA = 0.60


    def invoke(self, args):

        roi = args["final_img"]
        shape = args["shape"]
        cropped_image, _ = self.appCore.cropping(roi,shape,self.CROP_ALPHA)
        if cropped_image is None:
            return ("Brightness", -1)
        hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        avg = np.mean(v)
        return ("Brightness", avg)
