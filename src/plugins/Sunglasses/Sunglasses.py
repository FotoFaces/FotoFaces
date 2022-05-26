from logging import Logger

from engine import PluginCore
from model import Meta

import numpy as np
import cv2
import tensorflow as tf
import keras
from keras.models import model_from_json

class Sunglasses(PluginCore):

    def __init__(self, logger: Logger, appCore) -> None:
        super().__init__(logger, appCore)
        self.meta = Meta(
            name='Sunglasses plugin',
            description='Plugin for verifying if there are sunglasses',
            version='0.0.1'
        )
        self.appCore = appCore

    def invoke(self, args):

        image = args["image"]
        shape = args["shape"]

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Left eye
        if shape[40][0] != shape[41][0]:
            x_coords = (shape[40][0], shape[41][0])
            left_point = x_coords.index(min(x_coords))

            aux = shape[41] - shape[40]
            distance = np.linalg.norm(aux)
            h = int(distance)

            tl1 = (shape[40+left_point][0], shape[40+left_point][1])
            br1 = (shape[40+left_point][0]+h, shape[40+left_point][1]+h)
        else:
            tl1 = (shape[41][0], shape[41][1])
            br1 = (shape[41][0]+1, shape[41][1]+1)

        # Right eye
        if shape[46][0] != shape[47][0]:
            x_coords = (shape[46][0], shape[47][0])
            left_point = x_coords.index(min(x_coords))

            aux = shape[47] - shape[46]
            distance = np.linalg.norm(aux)
            h = int(distance)

            tl2 = (shape[46+left_point][0], shape[46+left_point][1])
            br2 = (shape[46+left_point][0]+h, shape[46+left_point][1]+h)
        else:
            tl2 = (shape[47][0], shape[47][1])
            br2 = (shape[47][0]+1, shape[47][1]+1)


        e1 = hsv_image[tl1[1]:br1[1],tl1[0]:br1[0]]

        e2 = hsv_image[tl2[1]:br2[1],tl2[0]:br2[0]]

        skin_reference = hsv_image[shape[30][1], shape[30][0]]
        h1, s1, v1 = cv2.split(e1)
        h2, s2, v2 = cv2.split(e2)

        S_average = (np.mean(s1)+np.mean(s2))/2
        V_average = (np.mean(v1)+np.mean(v2))/2

        S_diff = abs(skin_reference[1] - S_average)
        V_diff = abs(skin_reference[2] - V_average)

        if S_diff < 90 and V_diff < 90:
            return ("Sunglasses", detect_sunglassesNew(image, shape))
        else :
            return ("Sunglasses", True)
        #return ("Sunglasses", [S_diff, V_diff])

    def cropping_eye(image, shape):

        x1 = int(shape[0][0])
        x2 = int(shape[16][0])
        y1 = int(shape[19][1])
        y2 = int(shape[33][1])
        tl = (x1, y1)
        br = (x2, y2)

        if x1 >= 0 and y1 >= 0 and x2 < image.shape[1] and y2 < image.shape[0]:
            roi = image[tl[1]:br[1],tl[0]:br[0]]
            #cv2.imshow("Eye Region", roi)
            #cv2.waitKey()
            return roi
        else:
            return None

    def detect_sunglassesNew(image, shape):
        
        roi_eye = cropping_eye(image, shape) 
        img_size = 150
        resized_arr = cv2.resize(roi_eye, (img_size, img_size)) # Reshaping images to preferred size
        resized_arr = cv2.cvtColor(resized_arr, cv2.COLOR_RGB2BGR)

        x_test = []
        x_test.append(resized_arr)
        x_test = np.array(x_test)
        x_test = (x_test / 127.5 ) - 1	#Normalize the image channel values

        # load json and create model
        json_file = open('../model_tf_sunglasses_eyes_hand_final.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("../model_tf_sunglasses_eyes_hand_final.h5")

        # evaluate loaded model on test data
        loaded_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # Very low learning rate
                loss=keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[keras.metrics.BinaryAccuracy()])

        predictions = (loaded_model.predict(np.array(x_test)) > 0.5).astype("int32")

        print("Prediction Sunglasses: ", predictions[0])

        if predictions[0] == 1:
            return True

        return False
