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

    def cropping(image, shape, data):
        aux = shape[0] - shape[16]
        distance = np.linalg.norm(aux)

        h = int(distance)

        middle_X = int((shape[0][0] + shape[16][0])/2)
        middle_Y = int((shape[19][1] + shape[33][1])/2)

        x1 = int(middle_X-h*self.CROP_ALPHA)
        y1 = int(middle_Y-h*self.CROP_ALPHA)
        x2 = int(middle_X+h*self.CROP_ALPHA)
        y2 = int(middle_Y+h*self.CROP_ALPHA)
        tl = (x1, y1)
        br = (x2, y2)

        if x1 >= 0 and y1 >= 0 and x2 < image.shape[1] and y2 < image.shape[0]:
            roi = image[tl[1]:br[1],tl[0]:br[0]]
            return roi
        else:
            return None

    def invoke(self, args):

        roi = args["candidate"]
        shape = args["shape"]
        data = {}
        cropped_image = self.appCore.cropping(roi,shape,data )
        if cropped_image is None:
            return ("Brightness", -1)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        avg = np.mean(v)
        return [("Brightness", avg), ("Crop Position", data["Crop Position"])]
