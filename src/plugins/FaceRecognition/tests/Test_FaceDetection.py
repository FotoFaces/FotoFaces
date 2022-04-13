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


imagePath = sys.argv[1]
cascPath = sys.argv[2]  
faceCascade = cv2.CascadeClassifier(cascPath)
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# faces = faceCascade.detectMultiScale(
#     gray,
#     scaleFactor=1.1,
#     minNeighbors=5,
#     minSize=(30, 30),
#     flags = cv2.CASCADE_SCALE_IMAGE
# )

# print("Found {0} faces!".format(len(faces)))
start = time.time()
max_shape, max_bb, raw_shape = fotofaces.detect_face(gray)
end = time.time() - start
# Draw a rectangle around the faces
# for (x, y, w, h) in faces:
#     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.rectangle(image, (max_bb[0], max_bb[1]), (max_bb[0]+max_bb[2], max_bb[1]+max_bb[3]), (0, 255, 0), 2)

height, width = image.shape[:2]
print("Do gajo")
print(end)

start = time.time()
# faces = faceCascade.detectMultiScale(
#     gray,
#     scaleFactor=1.1,
#     minNeighbors=5,
#     minSize=(30, 30),
#     flags = cv2.CASCADE_SCALE_IMAGE
# )
end = time.time() - start
# for (x, y, w, h) in faces:
#     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
print("Nosso")
#print("Found {0} faces!".format(len(faces)))

print(end)


#print(height//4)
#print(width//4)

res = cv2.resize(image,(width//4, height//4), interpolation = cv2.INTER_AREA)
cv2.imshow("Faces found", res)
cv2.waitKey(0)


    