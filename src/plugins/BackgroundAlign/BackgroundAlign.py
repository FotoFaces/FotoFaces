from logging import Logger

from engine import PluginCore
from model import Meta

import numpy as np
import cv2

class BackgroundAlign(PluginCore):

    def __init__(self, logger: Logger, appCore) -> None:
        super().__init__(logger, appCore)
        self.meta = Meta(
            name='BackgroundAlign plugin',
            description='Plugin to fix the person face orientation',
            version='0.0.1'
        )
        self.appCore = appCore

    def invoke(self, args):

        image = args["final_img"]
        shape = args["shape"]

        rotated, shape = self.appCore.rotate(image, shape)

        blur_rotated = cv2.GaussianBlur(rotated, (3, 3), 0)

        gray = cv2.cvtColor(blur_rotated, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray,219,220)

        lines = cv2.HoughLines(edges, 1, np.pi / 180, 75, None, 0, 0)

        
        if len(lines) > 0:
            n_lines = len(lines)	#numero de linhas detetadas
            n_horizontal = 0
            n_vertical = 0
            n_rest = 0

            aux_image = blur_rotated

            for i in range(0,len(lines)):
                for rho,theta in lines[i]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))

                    if y1 != 0 and y2 != 0 and x1 != 0 and x2 != 0 and (y1 - y2) != 0 and (x1 - x2) != 0:
                        slope_as_angle = math.atan((y1 - y2) / (x1 - x2))
                        slope_as_angle = math.degrees(math.atan2((y1 - y2), (x1 - x2)))


                        limit = 10		# dez graus de liberdade para dizer que a linha é horizontal

                        if check_the_line(slope_as_angle, limit=limit) == "h":
                            n_horizontal = n_horizontal + 1
                        elif check_the_line(slope_as_angle, limit=limit) == "v":
                            n_vertical = n_vertical + 1
                        else:
                            n_rest = n_rest + 1


        if (n_vertical > n_rest) or (n_horizontal > n_rest) or (n_horizontal + n_vertical > n_rest): 
            return ("BackgroundAlign", rotated)
        else:
            return ("BackgroundAlign", None)




