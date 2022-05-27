from logging import Logger

from engine import PluginCore
from model import Meta

import numpy as np
import cv2

class FixFaceOrientation(PluginCore):

    def __init__(self, logger: Logger, appCore) -> None:
        super().__init__(logger, appCore)
        self.meta = Meta(
            name='FixFaceOrientation plugin',
            description='Plugin to fix the person face orientation',
            version='0.0.1'
        )
        self.appCore = appCore

    def invoke(self, args):

        image = args["final_img"]
        shape = args["shape"]

        image, shape = self.appCore.rotate(image, shape)

        image = cv2.GaussianBlur(image, (3, 3), 0)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray,219,220)

        #HOUGH LINE TRANSFORM OPENCV 

        #		---Probabilistic Hough Line---
        #img = image 
        #minLineLength = 100
        #maxLineGap = 1
        #lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
        
        #print("Linhas:")
        #print(lines[0])
        
        #for i in range(0,len(lines)):
        #	for x1,y1,x2,y2 in lines[i]:
        #		cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

        #cv2.imshow("Hough Line Transform", img)


        #		---"Normal" Hough Line---
        #lines = cv2.HoughLines(edges,1,np.pi/180,120)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 75, None, 0, 0)

        #print("Linhas:")
        #print(len(lines))
        
        if len(lines) > 0:
            n_lines = len(lines)	#numero de linhas detetadas
            n_horizontal = 0
            n_vertical = 0
            n_rest = 0

            aux_image = image

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

                    #cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
                    if y1 != 0 and y2 != 0 and x1 != 0 and x2 != 0 and (y1 - y2) != 0 and (x1 - x2) != 0:
                        slope_as_angle = math.atan((y1 - y2) / (x1 - x2))
                        slope_as_angle = math.degrees(math.atan2((y1 - y2), (x1 - x2)))

                        #print("Slope in degrees: ", slope_as_angle)
                        #print("----------------------")

                        limit = 10		# dez graus de liberdade para dizer que a linha é horizontal

                        if check_the_line(slope_as_angle, limit=limit) == "h":
                            n_horizontal = n_horizontal + 1
                            cv2.line(aux_image,(x1,y1),(x2,y2),(0,255,0),2)		#pintar as linahas horizontais 
                        elif check_the_line(slope_as_angle, limit=limit) == "v":
                            n_vertical = n_vertical + 1
                            cv2.line(aux_image,(x1,y1),(x2,y2),(255,0,0),2)		#pintar as linhas verticais 
                        else:
                            n_rest = n_rest + 1
                            cv2.line(aux_image,(x1,y1),(x2,y2),(0,0,255),2)		#pintar as linhas verticais 

            print("Horizontal lines number:", n_horizontal)
            print("Vertical lines number:", n_vertical)
            print("Rest lines number:", n_rest)

            cv2.imshow("Hough Line", aux_image)		#ver a imagem auxiliar com as linhas verticais e horizontais 
            cv2.waitKey(0)

        if (n_vertical > n_rest) or (n_horizontal > n_rest) or (n_horizontal + n_vertical > n_rest): 
            return "Good Background"
        else:
            return "Tilted Background"




        return ("Sunglasses", [S_diff, V_diff])
