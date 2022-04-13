from logging import Logger

from engine import PluginCore
from model import Meta

import numpy as np
import cv2
import math
import dlib
import base64
import json

CROP_ALPHA = 0.95

# loads everything needed
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

class FaceRecognition(PluginCore):

    def __init__(self, logger: Logger) -> None:
        super().__init__(logger)
        self.meta = Meta(
            name='Face Recognition plugin',
            description='Plugin that returns a distance between two faces within a roi, the bigger the distance the lower the people look a like.Meaning they are not the same person.',
            version='0.0.1'
        )

    def shape_to_np(self,shape):
        landmarks = np.zeros((68,2), dtype = int)
        for i in range(0,68):
            landmarks[i] = (shape.part(i).x, shape.part(i).y)
        return landmarks

    def detect_face(self, gray_image):
        rects = detector(gray_image, 1)
        max_area, max_bb, max_shape, raw_shape = (0, None, None, None)

        for (z, rect) in enumerate(rects):
            if rect is not None and rect.top() >= 0 and rect.right() < gray_image.shape[1] and rect.bottom() < gray_image.shape[0] and rect.left() >= 0:
                predicted = predictor(gray_image, rect)
                bb = self.rect_to_bb(rect)
                area = self.bb_area(bb)
                # only returns the largest bounding box to avoid smaller false positives
                if area > max_area:
                    max_area = area
                    max_bb = bb
                    max_shape = self.shape_to_np(predicted)
                    raw_shape = predicted

        return max_shape, max_bb, raw_shape


    def bb_area(self, bb):
        return (bb[0]+bb[2])*(bb[1]+bb[3])

    def rect_to_bb(self, rect):
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        return [x, y, w, h]


    def invoke(self, args):
        frame = args["candidate"]
        shape = args["raw_shape"]
        reference = args["reference"]

        candidate = dlib.get_face_chip(frame, shape)
        candidate_descriptor = facerec.compute_face_descriptor(candidate)
        # reads the PACO picture
        gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        shape, bb, raw_shape = self.detect_face(gray)
        if bb is not None:
            old = dlib.get_face_chip(reference, raw_shape)
            old_descriptor = facerec.compute_face_descriptor(old)

            candidate_descriptor = np.asarray(candidate_descriptor)
            old_descriptor = np.asarray(old_descriptor)
            dist = np.linalg.norm(old_descriptor - candidate_descriptor)
            self._logger.debug(f'Command: {args} -> {self.meta}')
            return [ ("Face Verification", dist),
                     ("Face Reference Detected", False) ]
        return ("Face Reference Detected", False)
