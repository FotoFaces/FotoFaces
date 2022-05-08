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

    def invoke(self, args):
        roi = args["candidate"]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        avg = np.mean(v)
        return ("Brightness", avg)
