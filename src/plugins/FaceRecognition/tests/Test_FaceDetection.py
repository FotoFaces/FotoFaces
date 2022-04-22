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
import face_recognition as fc



# (Assim por miudos) tradução de css para openCV 
def rect_to_bb(face):
	x = face[3]
	y = face[0]
	w = face[1] - x
	h = face[2] - y
	return [x, y, w, h]


###########     Face_recognition library
def testFaceRecognitionLib():
    print("Testing with Face_recognition library ....")
    im = cv2.imread(sys.argv[1]) #read image OpenCV para dps poder mexer nela

    image = fc.load_image_file(sys.argv[1]) #read image pela fc algorithm
    
    start = time.time()
    face_locations = fc.face_locations(image) # face detection algorithm
    duration = time.time() - start

    print(face_locations)

    for face in face_locations:
        [x,y,w,h] = rect_to_bb(list(face)) # conversão (só aceita ¯\_(ツ)_/¯ )
        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2) # desenhar retangulos
    
    cv2.imshow("Faces found", im)
    cv2.waitKey(0)
    print("Duration: "+ str(duration))
    print("End of Testing with OpenCV")


###########     Using OpenCV library

def testOpenCVDetection():
    print("Testing with OpenCV ....")
    
    start = time.time()
    image = OpenCVDetection(sys.argv[1]) 
    duration = time.time() - start
    
    height, width = image.shape[:2]
    res = cv2.resize(image,(width//4, height//4), interpolation = cv2.INTER_AREA)
    cv2.imshow("Faces found", res)
    cv2.waitKey(0)
    
    print("Duration: "+ str(duration))
    print("End of Testing with OpenCV")




def OpenCVDetection(imagePath):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #cena para detetar caras (tbm n sei muito os detalhes)
    
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #conversão para escala de cinzentos

    faces = faceCascade.detectMultiScale(   #algoritmo de detetação
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    ) 

    for (x, y, w, h) in faces: #desenhar caras
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image


###########     Using Old Fotofaces algorithm 

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
    #desenhar retangulos 
    cv2.rectangle(image, (max_bb[0], max_bb[1]), (max_bb[0]+max_bb[2], max_bb[1]+max_bb[3]), (0, 255, 0), 2)
    return image



###########     main 

testOpenCVDetection()
print("\n\n")
testOldDetection()
print("\n\n")
testFaceRecognitionLib()