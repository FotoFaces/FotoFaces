import dlib
import cv2
import math
import numpy as np
from logging import Logger
from engine import PluginCore
from model import Meta


class FaceRecognition(PluginCore):

    def __init__(self, logger: Logger, appCore) -> None:
        super().__init__(logger, appCore)
        self.meta = Meta(
            name='Face Recognition plugin',
            description='Plugin that returns a distance between two faces within a roi, the bigger the distance the lower the people look a like.Meaning they are not the same person.',
            version='0.0.1'
        )
        self.facerec = dlib.face_recognition_model_v1("../../../dlib_face_recognition_resnet_model_v1.dat")
        self.appCore = appCore


    def invoke(self, args):

        """
            Logic of the plugin
            :args is a dictionaire
            :returns a a value related to the metric analysed
        """

        #ficheiro para comparação de caras
        reference = args["reference"]
        candidate = args["candidate"]

        self._logger.info(args.keys())
        self._logger.info(candidate[:30])
        self._logger.info(reference[:30])




        photo_raw_shape = args["raw_shape"]

        #deteção de uma cara nas fotos
        reference_raw_shape = self.appCore.detect_face(reference)[2]


        #buscar o chip da cara
        reference_chip = dlib.get_face_chip(reference, reference_raw_shape)
        photo_chip = dlib.get_face_chip(candidate, photo_raw_shape)

        #usar face recognition file
        reference_descriptor = np.asarray(self.facerec.compute_face_descriptor(reference_chip))
        photo_descriptor = np.asarray(self.facerec.compute_face_descriptor(photo_chip))

        #resultado da comparação se for inferior a 0.6 são a mesma pessoa
        tolerance = np.linalg.norm(reference_descriptor - photo_descriptor)
        return ("Face Recognition", tolerance)
