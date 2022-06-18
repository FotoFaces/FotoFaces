from logging import Logger

from engine import PluginCore
from model import Meta

import tensorflow as tf
from keras.models import model_from_json
import keras
import numpy as np
import cv2
import os

class HatDetection(PluginCore):

    def __init__(self, logger: Logger, appCore) -> None:
        super().__init__(logger, appCore)
        self.meta = Meta(
            name='HatDetection plugin',
            description='Plugin to detect if a person is wearing a hat',
            version='0.0.1'
        )
        self.appCore = appCore
        self.net = cv2.dnn.readNet("./plugins/Hats/deploy.prototxt", "./plugins/Hats/hed_pretrained_bsds.caffemodel")
    # Detects sunglasses using  Machine Learning trained model

    def invoke(self, args):

        candidate = args["candidate"]
        shape = args["shape"]

        roi_cropp = self.cropping_hats(candidate, shape)
        #print("roi_cropp",roi_cropp)
        img_size = 150
        resized_arr = cv2.resize(roi_cropp, (img_size, img_size)) # Reshaping images to preferred size
        resized_arr = cv2.cvtColor(resized_arr, cv2.COLOR_RGB2BGR)

        x_test = []
        x_test.append(resized_arr)
        x_test = np.array(x_test)
        x_test = (x_test / 127.5 ) - 1

        # load json and create model
        json_file = open('./plugins/Hats/model_tf_hats_head_cropp_hand_final.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        print(os.getcwd())
        loaded_model.load_weights("./plugins/Hats/model_tf_hats_head_cropp_hand_final.h5")
        print("Loaded model from disk")

        # evaluate loaded model on test data
        loaded_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # Very low learning rate
                    loss=keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=[keras.metrics.BinaryAccuracy()])

        predictions = (loaded_model.predict(np.array(x_test)) > 0.5).astype("int32")

        #print("Prediction Hats: ", predictions[0])

        if predictions[0] == 1:
            return ('Hats' , "true")

        return ('Hats' , "false")


    # Crops the image into the head reagion to give as input to the hats trained model
    def cropping_hats(self,image, shape):

        CROP_ALPHA = 0.75

        aux = shape[0] - shape[16]
        distance = np.linalg.norm(aux)
        h = int(distance)

        middle_X = int((shape[0][0] + shape[16][0])/2)
        middle_Y = int((shape[19][1] + shape[29][1])/2)

        x1 = int((shape[0][0]))
        y1 = int(middle_Y-h*CROP_ALPHA)
        x2 = int((shape[16][0]))
        y2 = int((shape[19][1]))
        tl = (x1, y1)
        br = (x2, y2)

        if x1 >= 0 and y1 >= 0 and x2 < image.shape[1] and y2 < image.shape[0]:
            roi = image[tl[1]:br[1],tl[0]:br[0]]
            return roi

        else:
            return None
