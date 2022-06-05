import base64
import json
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg  
#from PIL import Image
import io     
import cv2

import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import pytest

api = 'http://192.168.1.69:5000/'
#reference = sys.argv[1]
#candidate = sys.argv[2]


def call_fotofaces(reference, candidate):

    with open(reference, "rb") as f:
        ref_bytes = f.read()
    ref_b64 = base64.b64encode(ref_bytes).decode("utf8")

    with open(candidate, "rb") as f:
        can_bytes = f.read()
    can_b64 = base64.b64encode(can_bytes).decode("utf8")

    mp_encoder = MultipartEncoder(
        fields={
            'candidate': can_b64 ,
            'id': '98083',
            'reference':ref_b64,
        }
    )
    r = requests.post(
        api,
        data=mp_encoder,  # The MultipartEncoder is posted as data, don't use files=...!
        # The MultipartEncoder provides the content-type header with the boundary:
        headers={'Content-Type': mp_encoder.content_type}
    )
    try:
        data = r.json()
        #print(data['feedback']) 
        return json.loads(data['feedback'])              
    except requests.exceptions.RequestException:
        print(r.text)
    return None

def  test_bright_face():
    reference = "images/Pedro.jpg"
    candidate = "images/bright_Pedro_1.jpg"
    results = call_fotofaces(reference,candidate)

    assert results["Colored Picture"] == "true"
    assert results["Face Candidate Detected"] == "true"
    assert results["Cropping"] == "true"
    assert results["Glasses"] == "false"
    assert results["Sunglasses"] == "false"
    assert results["Head Pose"][0] < 15 and results["Head Pose"][1] < 15 and results["Head Pose"][2] < 15 
    assert results["Eyes Open"] > 0.21
    assert results["focus"] > 85
    assert results["Face Recognition"] < 0.6
    assert results["Image Quality"] < 25
    assert results["Hats"] == "false"
    assert results["Brightness"] > 100





def  test_crop_fail():
    reference = "images/Pedro.jpg"
    candidate = "images/bright_vicente_6.jpg"
    results = call_fotofaces(reference,candidate)

    assert results["Colored Picture"] == "true"
    assert results["Face Candidate Detected"] == "true"
    assert results["Cropping"] == "false"


def  test_Glasses():
    reference = "images/Pedro.jpg"
    candidate = "images/Glasses_2.jpg"
    results = call_fotofaces(reference,candidate)

    assert results["Colored Picture"] == "true"
    assert results["Face Candidate Detected"] == "true"
    assert results["Cropping"] == "true"
    assert results["Glasses"] == "true"
    assert results["Sunglasses"] == "false"
    assert results["Head Pose"][0] < 15 and results["Head Pose"][1] < 15 and results["Head Pose"][2] < 15 
    assert results["Eyes Open"] > 0.21
    assert results["focus"] < 85
    assert results["Face Recognition"] > 0.6
    assert results["Image Quality"] < 25
    assert results["Hats"] == "false"
    assert results["Brightness"] > 100





# tested 
# Glasses
# focus
# Face recognition
