from flask import request, Flask
from engine import PluginEngine
from util import FileSystem
import requests
import logging

import numpy as np
import cv2
import math
import dlib
import base64
import json

import appCore
from flask_cors import CORS, cross_origin

# kafka implementation
import json
import sys
from random import choice
from argparse import ArgumentParser, FileType
from configparser import ConfigParser


app = Flask(__name__)
coreApplication = appCore.ApplicationCore()
cors = CORS(app, resources={r"/*": {"origins": "*"}})


logging.basicConfig(filename = "logs/logfile.log",
                    filemode = "w")

logger = logging.getLogger()

"""
Reads two input images and an identifier:
- Candidate picture
- PACO reference picture

Outputs a dictionary with the cropped image (depending on some requirements), metrics and the same identifier.
"""


# Cropping threshold (for higher values the cropping might be bigger than the image itself
# which will make the app consider that the face is out of bounds)
#CROP_ALPHA = 0.95

# loads everything needed
#detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")


@app.route("/hello", methods=["POST"])
@cross_origin()
def someOther():
    app.logger.info(request)
    app.logger.info(request.json)
    app.logger.info(request.form)
    return "ola", 200


@app.route("/", methods=["POST"])
@cross_origin()
def upload_image():
    #logger.info("update")
    if "candidate" in request.form.keys() and "id" in request.form.keys():
        #logger.info("first if")

        candidate = request.form["candidate"]
        identifier = request.form["id"]
        identifier_decoded = identifier
        candidate = cv2.imdecode(
            np.frombuffer(base64.b64decode(candidate), np.uint8), cv2.IMREAD_COLOR
        )

        app.logger.info(f"Identifier {identifier} requested updated with candidate photo {candidate[:30]}")
        data = {}
        data["Colored Picture"] = coreApplication.is_gray(candidate)
        if data["Colored Picture"] == False:
            #logger.info("no colored picture")

            dict_data = {"id": identifier_decoded, "feedback": json.dumps(data)}
            app.logger.info(dict_data)
            return dict_data
        else:
            #logger.info("colored picture")
            # reads the candidate picture
            gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
            shape, bb, raw_shape = coreApplication.detect_face(gray)
            data["Face Candidate Detected"] = True
            if bb is None:
                #logger.info("no face")
                # No face detected
                data["Face Candidate Detected"] = False
                dict_data = {"id": identifier_decoded, "feedback": json.dumps(data)}
                app.logger.info(dict_data)
                return dict_data
            else:
                #logger.info("face detected")
                image, shape = coreApplication.rotate(candidate, shape)
                roi = coreApplication.cropping(image, shape,data)
                data["Cropping"] = True
                if roi is None:
                    #logger.info("no cropping")
                    # Face is not centered and/or too close to the camera
                    data["Cropping"] = False
                    dict_data = {"id": identifier_decoded, "feedback": json.dumps(data)}
                    app.logger.info(dict_data)
                    return dict_data
                else:
                    #logger.info("cropping")
                    data["Resize"] = 500 / roi.shape[0]
                    final_img = cv2.resize(roi, (500, 500))
                    # start plugins

                    # old method
                    # img2 = request.form["reference"]
                    # app.logger.info(f"image reference {img2}")

                    # new method with kafka

                    response = requests.get(f'http://api:8393/image/{identifier_decoded}')
                    response_json = response.json()
                    old_photo = response_json["photo"]
                    app.logger.info(f"Received photo {old_photo[:30]}")
                    reference = cv2.imdecode(
                        np.frombuffer(base64.b64decode(old_photo), np.uint8),
                        cv2.IMREAD_COLOR,
                    )

                    app.logger.info(f"Received photo decoded by cv2 {reference[:30]}")
                    resp = plEngine.start(
                        candidate=candidate,
                        reference=reference,
                        raw_shape=raw_shape,
                        image=image,
                        shape=shape,
                        final_img=final_img
                    )

                    app.logger.info(f"{resp}")
                    for k, v in resp.items():
                        data[k] = v
                    __print_plugins_end()
                    app.logger.info(data)

                    #logger.info(data)
                    return data
    return "", 204


def __print_plugins_end() -> None:
    app.logger.info("-----------------------------------")
    app.logger.info("Plugins are done")
    app.logger.info("End of execution")
    app.logger.info("-----------------------------------")


def __init_app(**args):
    return


plEngine = PluginEngine(
    options={
        "log_level": "DEBUG",
        "directory": "./plugins/",
        "coreApplication": coreApplication,
    },
)


if __name__ == "__main__":
    # Parse the command line.

    app.run(debug=True, host="0.0.0.0", port=5000)
