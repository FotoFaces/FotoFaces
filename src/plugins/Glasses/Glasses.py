from logging import Logger

from engine import PluginCore
from model import Meta

import numpy as np
import cv2
import dlib

class Glasses(PluginCore):

    def __init__(self, logger: Logger, appCore) -> None:
        super().__init__(logger, appCore)
        self.meta = Meta(
            name='Glasses plugin',
            description='Plugin for verifying the existence of glasses',
            version='0.0.1'
        )
        self.appCore = appCore

    def invoke(self, args):

        shape = args["shape"]
        image = args["image"]

        nose_bridge_x = []
        nose_bridge_y = []
        for i in [28,29,30,31,33,34,35]:
                nose_bridge_x.append(shape[i][0])
                nose_bridge_y.append(shape[i][1])

        x_min = min(nose_bridge_x)
        x_max = max(nose_bridge_x)
        y_min = shape[20][1]
        y_max = shape[31][1]
        img2 = image[y_min:y_max, x_min:x_max]

        img_blur = cv2.GaussianBlur(np.array(img2),(3,3), sigmaX=0, sigmaY=0)
        edges = cv2.Canny(image =img_blur, threshold1=100, threshold2=200)
        edges_center = edges.T[(int(len(edges.T)/2))]

        has_glasses = "false"
        if 255 in edges_center:
            has_glasses = "true"
        return ("Glasses", has_glasses)
