import numpy as np
import cv2
import math
import dlib
import base64
import json

class ApplicationCore():

    # Cropping threshold (for higher values the cropping might be bigger than the image itself
    # which will make the app consider that the face is out of bounds)
    def __init__(self):
        # loads everything needed
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Calculates the bounding box area
    def bb_area(self,bb):
        return (bb[0] + bb[2]) * (bb[1] + bb[3])


# Converts dlib format to numpy format
    def shape_to_np(self,shape):
        landmarks = np.zeros((68, 2), dtype=int)

        for i in range(0, 68):
            landmarks[i] = (shape.part(i).x, shape.part(i).y)

        return landmarks


# Converts dlib format to opencv format
    def rect_to_bb(self,rect):
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        return [x, y, w, h]


    def is_gray(self,img):
        b, g, r = cv2.split(img)
        if np.array_equal(b, g) and np.array_equal(b, r):
            return False
        return True


# Detects faces and only returns the largest bounding box
    def detect_face(self,image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #conversão para escala de cinzentos
        rects = self.detector(gray_image, 1)
        max_area, max_bb, max_shape, raw_shape = (0, None, None, None)

        for (z, rect) in enumerate(rects):
            if (
                rect is not None
                and rect.top() >= 0
                and rect.right() < gray_image.shape[1]
                and rect.bottom() < gray_image.shape[0]
                and rect.left() >= 0
            ):
                predicted = self.predictor(gray_image, rect)
                bb = self.rect_to_bb(rect)
                area = self.bb_area(bb)
                # only returns the largest bounding box to avoid smaller false positives
                if area > max_area:
                    max_area = area
                    max_bb = bb
                    max_shape = self.shape_to_np(predicted)
                    raw_shape = predicted

        return max_shape, max_bb, raw_shape


# Applies rotation correction
    def rotate(self,image, shape):
        dY = shape[36][1] - shape[45][1]
        dX = shape[36][0] - shape[45][0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        dst = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)

        # transform points
        ones = np.ones(shape=(len(shape), 1))
        points_ones = np.hstack([shape, ones])
        new_shape = M.dot(points_ones.T).T
        new_shape = new_shape.astype(int)

        return dst, new_shape


# Crops the image into the final PACO image
    def cropping(self,image, shape, crop_alpha=0.95):
        aux = shape[0] - shape[16]
        distance = np.linalg.norm(aux)

        h = int(distance)

        middle_X = int((shape[0][0] + shape[16][0]) / 2)
        middle_Y = int((shape[19][1] + shape[33][1]) / 2)

        x1 = int(middle_X - h *crop_alpha)
        y1 = int(middle_Y - h *crop_alpha)
        x2 = int(middle_X + h *crop_alpha)
        y2 = int(middle_Y + h *crop_alpha)
        tl = (x1, y1)
        br = (x2, y2)

        #if "Crop Position" not in data.keys():
        #    data["Crop Position"] = [x1, y1, x2, y2]
        if x1 >= 0 and y1 >= 0 and x2 < image.shape[1] and y2 < image.shape[0]:
            roi = image[tl[1] : br[1], tl[0] : br[0]]
            return roi
        else:
            return None
