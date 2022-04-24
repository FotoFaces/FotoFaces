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

    def __init__(self, logger: Logger, appCore) -> None:
        super().__init__(logger, appCore)
        self.meta = Meta(
            name='Face Recognition plugin',
            description='Plugin that returns a distance between two faces within a roi, the bigger the distance the lower the people look a like.Meaning they are not the same person.',
            version='0.0.1'
        )
        self.appCore = appCore


    def invoke(self, args):
        frame = args["candidate"]
        shape = args["raw_shape"]
        reference = args["reference"]

        candidate = dlib.get_face_chip(frame, shape)
        candidate_descriptor = facerec.compute_face_descriptor(candidate)
        # reads the PACO picture
        gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        shape, bb, raw_shape = self.appCore.detect_face(gray)
        if bb is not None:
            old = dlib.get_face_chip(reference, raw_shape)
            old_descriptor = facerec.compute_face_descriptor(old)

            candidate_descriptor = np.asarray(candidate_descriptor)
            old_descriptor = np.asarray(old_descriptor)
            dist = np.linalg.norm(old_descriptor - candidate_descriptor)
            self._logger.debug(f'Command: {args} -> {self.meta}')
            return [ ("Face Verification", dist),
                     ("Face Reference Detected", True) ]
        return ("Face Reference Detected", False)
