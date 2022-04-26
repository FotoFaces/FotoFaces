from flask import request, Flask
from engine import PluginEngine
from util import FileSystem

import numpy as np
import cv2
import math
import dlib
import base64
import json

# kafka implementation
import json
import sys
from random import choice
from argparse import ArgumentParser, FileType
from configparser import ConfigParser
from confluent_kafka import Producer, Consumer, OFFSET_BEGINNING


app = Flask(__name__)


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
    if args.reset:
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

# Calculates the bounding box area
def bb_area(bb):
    return (bb[0] + bb[2]) * (bb[1] + bb[3])


# Converts dlib format to numpy format
def shape_to_np(shape):
    landmarks = np.zeros((68, 2), dtype=int)

    for i in range(0, 68):
        landmarks[i] = (shape.part(i).x, shape.part(i).y)

    return landmarks


# Converts dlib format to opencv format
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return [x, y, w, h]


def is_gray(img):
    b, g, r = cv2.split(img)
    if np.array_equal(b, g) and np.array_equal(b, r):
        return False
    return True


# Detects faces and only returns the largest bounding box
def detect_face(gray_image):
    rects = detector(gray_image, 1)
    max_area, max_bb, max_shape, raw_shape = (0, None, None, None)

    for (z, rect) in enumerate(rects):
        if (
            rect is not None
            and rect.top() >= 0
            and rect.right() < gray_image.shape[1]
            and rect.bottom() < gray_image.shape[0]
            and rect.left() >= 0
        ):
            predicted = predictor(gray_image, rect)
            bb = rect_to_bb(rect)
            area = bb_area(bb)
            # only returns the largest bounding box to avoid smaller false positives
            if area > max_area:
                max_area = area
                max_bb = bb
                max_shape = shape_to_np(predicted)
                raw_shape = predicted

    return max_shape, max_bb, raw_shape


# Applies rotation correction
def rotate(image, shape):
    dY = shape[36][1] - shape[45][1]
    dX = shape[36][0] - shape[45][0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    dst = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    # transform points
    ones = np.ones(shape=(len(shape), 1))
    points_ones = np.hstack([shape, ones])
    new_shape = M.dot(points_ones.T).T
    new_shape = new_shape.astype(int)

    return dst, new_shape


# Crops the image into the final PACO image
def cropping(image, shape, data):
    aux = shape[0] - shape[16]
    distance = np.linalg.norm(aux)

    h = int(distance)

    middle_X = int((shape[0][0] + shape[16][0]) / 2)
    middle_Y = int((shape[19][1] + shape[33][1]) / 2)

    x1 = int(middle_X - h * CROP_ALPHA)
    y1 = int(middle_Y - h * CROP_ALPHA)
    x2 = int(middle_X + h * CROP_ALPHA)
    y2 = int(middle_Y + h * CROP_ALPHA)
    tl = (x1, y1)
    br = (x2, y2)

    data["Crop Position"] = (x1, y1, x2, y2)
    if x1 >= 0 and y1 >= 0 and x2 < image.shape[1] and y2 < image.shape[0]:
        roi = image[tl[1] : br[1], tl[0] : br[0]]
        return roi
    else:
        return None


@app.route("/", methods=["POST"])
def upload_image():
    if "candidate" in request.form.keys() and "id" in request.form.keys():
        img1 = request.form["candidate"]
        identifier = request.form["id"]
        identifier_decoded = identifier
        candidate = cv2.imdecode(
            np.frombuffer(base64.b64decode(img1), np.uint8), cv2.IMREAD_COLOR
        )
        
        
        # # Kafka Implementation to message deal with the REST API
        
        # # GET photo from the database
        # # produce a json message to send to the consumer
        # producer.produce(TOPIC_PRODUCE, json.dumps({"command": "get_photo", "id": identifier}))
        # producer.flush()
    
        # # Poll for new messages from Kafka and save the json object
        # msg_json = None
        # try:
        #     while True:
        #         msg = consumer.poll(1.0)
        #         if msg is None:
        #             pass
        #         elif msg.error():
        #             print(f"ERROR Recieving GET from the Database: {msg.error()}")
        #         else:
        #             msg_json = json.loads(msg.value().decode('utf-8'))
        #             print(f"Consumed event from topic {TOPIC_CONSUME}: message = {msg_json}")
        #             break
        # except KeyboardInterrupt:
        #     return False
        
        # # idk why it needs this but it doesn't work without it
        # msg_json = json.loads(msg_json)
        # # old photo from the database
        # old_photo = msg_json["photo"]
        # print(old_photo)
        
        
        
        data = {}
        data["Colored Picture"] = is_gray(candidate)
        if data["Colored Picture"] == False:
            dict_data = {"id": identifier_decoded, "feedback": json.dumps(data)}
            return dict_data
        else:
            # reads the candidate picture
            gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
            shape, bb, raw_shape = detect_face(gray)
            data["Face Candidate Detected"] = True
            if bb is None:
                # No face detected
                data["Face Candidate Detected"] = False
                dict_data = {"id": identifier_decoded, "feedback": json.dumps(data)}
                return dict_data
            else:
                image, shape = rotate(candidate, shape)
                roi = cropping(image, shape, data)
                data["Cropping"] = True
                if roi is None:
                    # Face is not centered and/or too close to the camera
                    data["Cropping"] = False
                    dict_data = {"id": identifier_decoded, "feedback": json.dumps(data)}
                    return dict_data
                else:
                    data["Resize"] = 500 / roi.shape[0]
                    final_img = cv2.resize(roi, (500, 500))
                    # start plugins
                    img2 = request.form["reference"]
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
                    print(f"{resp}")
                    for k,v in resp.items():
                        data[k] = v
                    __print_plugins_end()
                    return data
    return "", 204


def __print_plugins_end() -> None:
    print("-----------------------------------")
    print("Plugins are done")
    print("End of execution")
    print("-----------------------------------")


def __init_app(**args):
    return plEngine.start(**args)

plEngine = PluginEngine(options=
                        {"log_level": "DEBUG", "directory": "./plugins/"},
        )


if __name__ == "__main__":
    # Parse the command line.
    parser = ArgumentParser()
    parser.add_argument('config_file', type=FileType('r'))
    parser.add_argument('--reset', action='store_true')
    args = parser.parse_args()
    
    # Parse the configuration.
    # See https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
    config_parser = ConfigParser()
    config_parser.read_file(args.config_file)
    config = dict(config_parser['default'])
    
    # Create Producer instance
    producer = Producer(config)    

    # Create Consumer instance
    config.update(config_parser['consumer'])
    consumer = Consumer(config)

    consumer.subscribe([TOPIC_CONSUME], on_assign=reset_offset)
    
    app.run(debug=True)
