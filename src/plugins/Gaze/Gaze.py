from logging import Logger
from engine import PluginCore
from model import Meta
import numpy as np
import cv2


# Estimates gaze by detecting the pupils through morphological operations, finding their contours and estimating their distance to the corners
class GazePlugin(PluginCore):

    def __init__(self, logger: Logger) -> None:
        super().__init__(logger)
        self.meta = Meta(
            name='Gaze Plugin',
            description='Plugin for Gaze detection',
            version='0.0.1'
        )

    def dist_ratio(self, p1, p2, center):
        p1_dist = abs(center-p1)
        p2_dist = abs(center-p2)

        if p1_dist > p2_dist:
            ratio = (p2_dist/p1_dist)*100
        else:
            ratio = (p1_dist/p2_dist)*100

        return ratio


    def invoke(self, args):
        image = args["image"]
        shape = args["shape"]
    
        ratio = []
        kernel = np.ones((3, 3), np.uint8)
        # Left eye
        x1 = shape[37][0]
        y1 = int((shape[37][1]+shape[36][1])/2)
        x2 = int((shape[39][0]+shape[40][0])/2)
        y2 = int((shape[39][1]+shape[40][1])/2)

        tl1 = (min(x1, x2), min(y1, y2))
        br1 = (max(x1, x2), max(y1, y2))

        if tl1[1] != br1[1] and tl1[0] != br1[0]:
            e1 = image[tl1[1]:br1[1],tl1[0]:br1[0]]
            e1p = cv2.cvtColor(e1, cv2.COLOR_BGR2GRAY)
            e1p = cv2.bilateralFilter(e1p, 10, 30, 30)
            e1p = cv2.erode(e1p, kernel, iterations=3)
            ret, e1p = cv2.threshold(e1p,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            contours, hierarchy = cv2.findContours(e1p, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                contours = max(contours, key = cv2.contourArea)
                try:
                    moments = cv2.moments(contours)
                    x = int(moments["m10"] / moments["m00"])
                    y = int(moments["m01"] / moments["m00"])
                    e1_og = (tl1[0]+x, tl1[1]+y)
                    ratio1 = self.dist_ratio(shape[36][0], shape[39][0], e1_og[0])
                    ratio.append(ratio1)
                except (IndexError, ZeroDivisionError):
                    pass

        # Right eye
        x1 = shape[44][0]
        y1 = int((shape[44][1]+shape[45][1])/2)
        x2 = int((shape[42][0]+shape[47][0])/2)
        y2 = int((shape[42][1]+shape[47][1])/2)

        tl2 = (min(x1, x2), min(y1, y2))
        br2 = (max(x1, x2), max(y1, y2))

        if tl2[1] != br2[1] and tl2[0] != br2[0]:
            e2 = image[tl2[1]:br2[1],tl2[0]:br2[0]]

            e2p = cv2.cvtColor(e2, cv2.COLOR_BGR2GRAY)
            e2p = cv2.bilateralFilter(e2p, 10, 30, 30)
            e2p = cv2.erode(e2p, kernel, iterations=3)
            ret, e2p = cv2.threshold(e2p,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            contours, hierarchy = cv2.findContours(e2p, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                contours = max(contours, key = cv2.contourArea)
                try:
                    moments = cv2.moments(contours)
                    x = int(moments["m10"] / moments["m00"])
                    y = int(moments["m01"] / moments["m00"])
                    e2_og = (tl2[0]+x, tl2[1]+y)
                    ratio2 = self.dist_ratio(shape[42][0], shape[45][0], e2_og[0])
                    ratio.append(ratio2)
                except (IndexError, ZeroDivisionError):
                    pass

        focus = 0
        if len(ratio) == 2:
            focus = sum(ratio)/len(ratio)
        elif len(ratio) == 1:
            focus = ratio[0]
    
        return ("focus", focus)
