from logging import Logger

from engine import PluginCore
from model import Meta

import numpy as np
import cv2

class HatDetection(PluginCore):

    def __init__(self, logger: Logger, appCore) -> None:
        super().__init__(logger, appCore)
        self.meta = Meta(
            name='HatDetection plugin',
            description='Plugin to detect if a person is wearing a hat',
            version='0.0.1'
        )
        self.appCore = appCore
        self.net = cv2.dnn.readNet("deploy.prototxt", "hed_pretrained_bsds.caffemodel")

    def invoke(self, args):

        candidate = args["final_img"]
        shape = args["shape"]

        img_width = candidate.shape[1]
        img_height = candidate.shape[0]

        if img_width <= 1000 or  img_height <= 1000:
            scale_factor = 1
        if 1000 < img_width <= 2000 or 1000 < img_height <= 2000:
            scale_factor = 2
        if img_width > 2000 or img_height > 2000:
            scale_factor = 3

        candidate = cv2.resize(candidate, (int(img_width/scale_factor),int(img_height/scale_factor)))


        background = -1
            

        final_img, final_shape = self.appCore.rotate(candidate, shape)	#Face Alignment 

        if final_shape is not None:

            #cv2.imshow("Paco Image", final_img)

            #print("Image Face Shape: ", final_shape)

            inp = cv2.dnn.blobFromImage(final_img, scalefactor=1.0, size=(500, 500),
                            mean=(104.00698793, 116.66876762, 122.67891434),
                            swapRB=False, crop=False)
            
            self.net.setInput(inp)
            out = self.net.forward()
            out = out[0, 0]
            out = cv2.resize(out, (final_img.shape[1], final_img.shape[0]))
            out = 255 * out
            out = out.astype(np.uint8)
            out=cv2.cvtColor(out,cv2.COLOR_GRAY2BGR)
            con=np.concatenate((final_img,out),axis=1)

            background = background_rotation(final_img, final_shape, out)

            if background == 1: 
                return ("BackgroundAlign", final_img)
            else:
                return ("BackgroundAlign", None)


    def background_rotation(image, shape, img_edges):

        #global x 

        back = -1

        #Excluir a região da face da imagem para ficar apenas com o background 
        AUX = 0.95

        aux = shape[0] - shape[16]
        distance = np.linalg.norm(aux)
        h = int(distance)

        middle_X = int((shape[0][0] + shape[16][0])/2)
        middle_Y = int((shape[19][1] + shape[29][1])/2)

        x1 = int((shape[0][0])) - int(0.2*h)
        y1 = int(middle_Y-h*AUX)
        x2 = int((shape[16][0])) + int(0.2*h)
        y2 = int((shape[57][1]))

        #Excluir a região abaixo da linha da boca 
        x1_1 = int(0)
        y1_1 = int((shape[57][1]))
        x2_1 = int(image.shape[1])
        y2_1 = int(image.shape[0])
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #convert BGR to RGB
        image_blur = cv2.GaussianBlur(image, (5, 5), 0)

        gray = cv2.cvtColor(image_blur, cv2.COLOR_RGB2GRAY)


        img_edges_new = img_edges
        lines_P = find_lines_using_hough_lines_P(img_edges_new)

        if lines_P is not None:
            img_before = image
            median_angle_P = calculate_angle_P(lines_P, img_before)

            if -5 <= median_angle_P <= 5 or 85 <= median_angle_P <= 90 or -90 <= median_angle_P <= -85:
                back = 1	#Background direito se a média dos ângulos estiver entre estes valores 
            else:
                back = 0
            
        else:
            back = 1	#Background direito se não forem detetadas linhas

        return back 

    def find_lines_using_hough_lines_P(img_edges):
    
        img_edges = cv2.cvtColor(img_edges, cv2.COLOR_BGR2GRAY)
        lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=130, maxLineGap=5)

        return lines

    def calculate_angle_P(lines, img_before):
        angles = []
        for i in range(0,len(lines)):
            for x1,y1,x2,y2 in lines[i]:
    
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))  # find angle of line connecting (0,0) to (x,y) from +ve x axis
                angles.append(angle)

        median_angle = np.median(angles)
        return median_angle





