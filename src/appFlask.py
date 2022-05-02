from flask import request, Flask
from engine import PluginEngine
from util import FileSystem

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
from confluent_kafka import Producer, Consumer, OFFSET_BEGINNING


app = Flask(__name__)
coreApplication = appCore.ApplicationCore()
cors = CORS(app, resources={r"/*": {"origins": "*"}})


"""
Reads two input images and an identifier:
- Candidate picture
- PACO reference picture

Outputs a dictionary with the cropped image (depending on some requirements), metrics and the same identifier.
"""

# kafka implementation
# Topic for producing messages
TOPIC_PRODUCE = "image"
# Topic for consuming messages
TOPIC_CONSUME = "rev_image"

# Set up a callback to handle the '--reset' flag.
def reset_offset(consumer, partitions):
    if ARGS.reset:
        for p in partitions:
            p.offset = OFFSET_BEGINNING
        consumer.assign(partitions)


# Cropping threshold (for higher values the cropping might be bigger than the image itself
# which will make the app consider that the face is out of bounds)
CROP_ALPHA = 0.95

# loads everything needed
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")


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
    app.logger.info("recieved")
    if True:
        candidate = request.form["candidate"]
        identifier = request.form["id"]
        identifier_decoded = identifier
        candidate = cv2.imdecode(
            np.frombuffer(base64.b64decode(candidate), np.uint8), cv2.IMREAD_COLOR
        )

        # Kafka Implementation to message deal with the REST API

        # Parse the configuration.
        # See https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
        config_parser.read_file(ARGS.config_file)
        config = dict(config_parser["default"])

        # Create Producer instance
        producer = Producer(config)

        # Create Consumer instance
        config.update(config_parser["consumer"])
        consumer = Consumer(config)

        consumer.subscribe([TOPIC_CONSUME], on_assign=reset_offset)

        # GET photo from the database
        # produce a json message to send to the consumer
        producer.produce(
            TOPIC_PRODUCE, json.dumps({"command": "get_photo", "id": identifier})
        )
        producer.flush()
        app.logger.info('SENT: {"command": "get_photo", "id": ' + identifier + "}")

        # Poll for new messages from Kafka and save the json object
        msg_json = None
        try:
            while True:
                app.logger.info("here")
                msg = consumer.poll(1.0)
                if msg is None:
                    app.logger.info("None")
                    pass
                elif msg.error():
                    app.logger.info(f"ERROR Recieving GET from the Database: {msg.error()}")
                else:
                    msg_json = json.loads(msg.value().decode("utf-8"))
                    app.logger.info(f"Consumed event from topic {TOPIC_CONSUME}")
                    break
        except KeyboardInterrupt:
            return False

        # idk why it needs this but it doesn't work without it
        msg_json = json.loads(msg_json)
        # old photo from the database
        old_photo = msg_json["photo"]
        app.logger.info(msg_json)

        data = {}
        data["Colored Picture"] = coreApplication.is_gray(candidate)
        if data["Colored Picture"] == False:
            dict_data = {"id": identifier_decoded, "feedback": json.dumps(data)}
            app.logger.info(dict_data)
            return dict_data
        else:
            # reads the candidate picture
            gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
            shape, bb, raw_shape = coreApplication.detect_face(gray)
            data["Face Candidate Detected"] = True
            if bb is None:
                # No face detected
                data["Face Candidate Detected"] = False
                dict_data = {"id": identifier_decoded, "feedback": json.dumps(data)}
                app.logger.info(dict_data)
                return dict_data
            else:
                image, shape = coreApplication.rotate(candidate, shape)
                roi = coreApplication.cropping(image, shape, data)
                data["Cropping"] = True
                if roi is None:
                    # Face is not centered and/or too close to the camera
                    data["Cropping"] = False
                    dict_data = {"id": identifier_decoded, "feedback": json.dumps(data)}
                    app.logger.info(dict_data)
                    return dict_data
                else:
                    data["Resize"] = 500 / roi.shape[0]
                    final_img = cv2.resize(roi, (500, 500))
                    # start plugins

                    # old method
                    # img2 = request.form["reference"]
                    # app.logger.info(f"image reference {img2}")

                    # new method with kafka
                    img2 = old_photo

                    reference = cv2.imdecode(
                        np.frombuffer(base64.b64decode(img2), np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    resp = __init_app(
                        candidate=candidate,
                        reference=reference,
                        raw_shape=raw_shape,
                        image=image,
                        shape=shape,
                        final_img=final_img,
                    )

                    app.logger.info(f"{resp}")
                    for k, v in resp.items():
                        data[k] = v
                    __print_plugins_end()
                    app.logger.info(data)
                    return data
    return "", 204


def __print_plugins_end() -> None:
    app.logger.info("-----------------------------------")
    app.logger.info("Plugins are done")
    app.logger.info("End of execution")
    app.logger.info("-----------------------------------")


def __init_app(**args):
    return plEngine.start(**args)


plEngine = PluginEngine(
    options={
        "log_level": "DEBUG",
        "directory": "./plugins/",
        "coreApplication": coreApplication,
    },
)


if __name__ == "__main__":
    # Parse the command line.
    parser = ArgumentParser()
    parser.add_argument('config_file', type=FileType('r'))
    parser.add_argument('--reset', action='store_true')
    ARGS = parser.parse_args()
    config_parser = ConfigParser()

    app.run(debug=True, host="0.0.0.0", port=5000)
