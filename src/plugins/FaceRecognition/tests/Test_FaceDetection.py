import pytest
from flask import request, Flask
import numpy as np
import cv2
import math
import dlib
import base64
import json
import sys
import fotofaces
import time
#import FaceRecognition







def testOpenCVDetection():
    print("Testing with OpenCV ....")
    start = time.time()
    image = OpenCVDetection(sys.argv[1], sys.argv[2])
    duration = time.time() - start
    height, width = image.shape[:2]
    res = cv2.resize(image,(width//4, height//4), interpolation = cv2.INTER_AREA)
    cv2.imshow("Faces found", res)
    cv2.waitKey(0)
    print("Duration: "+ str(duration))
    print("End of Testing with OpenCV")




def OpenCVDetection(imagePath, cascPath):
    faceCascade = cv2.CascadeClassifier(cascPath)
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image



def testOldDetection():
    print("Testing with Old Fotofaces Algorithm ....")
    start = time.time()
    image = OldDetection(sys.argv[1])
    duration = time.time() - start
    height, width = image.shape[:2]
    res = cv2.resize(image,(width//4, height//4), interpolation = cv2.INTER_AREA)
    cv2.waitKey(0)
    print("Duration: "+ str(duration))
    print("End of Testing with Old Fotofaces Algorithm")

def OldDetection(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    max_shape, max_bb, raw_shape = fotofaces.detect_face(gray)
    cv2.rectangle(image, (max_bb[0], max_bb[1]), (max_bb[0]+max_bb[2], max_bb[1]+max_bb[3]), (0, 255, 0), 2)
    return image

testOpenCVDetection()
print("\n\n")
testOldDetection()