from logging import Logger

from engine import PluginCore
from model import Meta

import numpy as np
import cv2

class EyesOpen(PluginCore):

    def __init__(self, logger: Logger) -> None:
        super().__init__(logger)
        self.meta = Meta(
            name='Eyes Open plugin',
            description='Plugin for verifying if the eyes are open',
            version='0.0.1'
        )
        
    def invoke(self, args):
        shape = args["shape"]
        
        leftEAR = (np.linalg.norm(shape[37]-shape[41]) + np.linalg.norm(shape[38]-shape[40]))/(2*np.linalg.norm(shape[36]-shape[39]))
        rightEAR = (np.linalg.norm(shape[43]-shape[47]) + np.linalg.norm(shape[44]-shape[46]))/(2*np.linalg.norm(shape[42]-shape[45]))

        avg = (leftEAR+rightEAR)/2
        
        return ("Eyes Open", avg)
