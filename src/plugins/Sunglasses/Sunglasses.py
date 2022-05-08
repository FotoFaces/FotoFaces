from logging import Logger

from engine import PluginCore
from model import Meta

import numpy as np
import cv2

class Sunglasses(PluginCore):

    def __init__(self, logger: Logger, appCore) -> None:
        super().__init__(logger, appCore)
        self.meta = Meta(
            name='Sunglasses plugin',
            description='Plugin for verifying if there are sunglasses',
            version='0.0.1'
        )
        self.appCore = appCore

    def invoke(self, args):

        image = args["image"]
        shape = args["shape"]

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Left eye
        if shape[40][0] != shape[41][0]:
            x_coords = (shape[40][0], shape[41][0])
            left_point = x_coords.index(min(x_coords))

            aux = shape[41] - shape[40]
            distance = np.linalg.norm(aux)
            h = int(distance)

            tl1 = (shape[40+left_point][0], shape[40+left_point][1])
            br1 = (shape[40+left_point][0]+h, shape[40+left_point][1]+h)
        else:
            tl1 = (shape[41][0], shape[41][1])
            br1 = (shape[41][0]+1, shape[41][1]+1)

        # Right eye
        if shape[46][0] != shape[47][0]:
            x_coords = (shape[46][0], shape[47][0])
            left_point = x_coords.index(min(x_coords))

            aux = shape[47] - shape[46]
            distance = np.linalg.norm(aux)
            h = int(distance)

            tl2 = (shape[46+left_point][0], shape[46+left_point][1])
            br2 = (shape[46+left_point][0]+h, shape[46+left_point][1]+h)
        else:
            tl2 = (shape[47][0], shape[47][1])
            br2 = (shape[47][0]+1, shape[47][1]+1)


        e1 = hsv_image[tl1[1]:br1[1],tl1[0]:br1[0]]

        e2 = hsv_image[tl2[1]:br2[1],tl2[0]:br2[0]]

        skin_reference = hsv_image[shape[30][1], shape[30][0]]
        h1, s1, v1 = cv2.split(e1)
        h2, s2, v2 = cv2.split(e2)

        S_average = (np.mean(s1)+np.mean(s2))/2
        V_average = (np.mean(v1)+np.mean(v2))/2

        S_diff = abs(skin_reference[1] - S_average)
        V_diff = abs(skin_reference[2] - V_average)


        return ("Sunglasses", [S_diff, V_diff])
