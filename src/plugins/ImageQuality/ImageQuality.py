from logging import Logger

from engine import PluginCore
from model import Meta

import cv2

class ImageQuality(PluginCore):

    def __init__(self, logger: Logger, appCore) -> None:
        super().__init__(logger, appCore)
        self.meta = Meta(
            name='Image Quality plugin',
            description='Plugin for quality level of the image',
            version='0.0.1'
        )
        self.appCore = appCore

    def invoke(self, args):
        roi = args["final_img"]

        img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        score = cv2.quality.QualityBRISQUE_compute(img, "brisque_model_live.yml", "brisque_range_live.yml")
        obj = cv2.quality.QualityBRISQUE_create("brisque_model_live.yml", "brisque_range_live.yml")
        score = obj.compute(img)[0]

        return ("Image Quality", score)
