from logging import Logger

from engine import PluginCore
from model import Meta

import math

import numpy as np
import cv2

class HeadPose(PluginCore):

    def __init__(self, logger: Logger, appCore) -> None:
        super().__init__(logger, appCore)
        self.meta = Meta(
            name='Head pose plugin',
            description='Plugin for evaluate the pose of the head',
            version='0.0.1'
        )

        self.appCore = appCore

    def invoke(self, args):

        im = args["image"]
        features = args["shape"]

        size = im.shape

        image_points = np.array([
                                features[30],     # Nose
                                features[36],     # Left eye
                                features[45],     # Right eye
                                features[48],     # Left Mouth corner
                                features[54],     # Right mouth corner
                                features[8]		  # Chin
                                ], dtype="double")

        # 3D model points.
        model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose
                                (-165.0, 170.0, -135.0),     # Left eye
                                (165.0, 170.0, -135.0),      # Right eye
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0),     # Right mouth corner
                                (0.0, -330.0, -65.0)		 # Chin
                                ])

        # Camera internals
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
                                [[focal_length, 0, center[0]],
                                [0, focal_length, center[1]],
                                [0, 0, 1]], dtype = "double"
                                )

        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        rv_matrix = cv2.Rodrigues(rotation_vector)[0]

        proj_matrix = np.hstack((rv_matrix, translation_vector))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        pitch, yaw, roll = [math.radians(x) for x in eulerAngles]
        pitch = abs(math.degrees(math.asin(math.sin(pitch))))
        roll = abs(math.degrees(math.asin(math.sin(roll))))
        yaw = abs(math.degrees(math.asin(math.sin(yaw))))

        return ("Head Pose", [pitch, roll, yaw])
