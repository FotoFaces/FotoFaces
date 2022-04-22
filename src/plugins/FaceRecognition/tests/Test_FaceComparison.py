import face_recognition as fc
import cv2
import sys
import fotofaces
import time
import dlib
import numpy as np



facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	return [x, y, w, h]

def bb_area(bb):
	return (bb[0]+bb[2])*(bb[1]+bb[3])

def shape_to_np(shape):
	landmarks = np.zeros((68,2), dtype = int)

	for i in range(0,68):
		landmarks[i] = (shape.part(i).x, shape.part(i).y)

	return landmarks

def dlib_to_shape(face):
	x = face[3]
	y = face[0]
	w = face[1] - x
	h = face[2] - y
	return [x, y, w, h]


def main():
    print("Testing with Old Fotofaces Algorithm ....")

    start = time.time()

    result = testOldFotofaces()
    duration = time.time() - start

    print("Result:" + str(result))
    print("Duration: "+ str(duration))
    print("End of Testing with Old Fotofaces Algorithm")

    print("\n\n")

    print("Testing with Face Recognition lib ....")

    start = time.time()

    result = testFaceRecognitionLib()
    duration = time.time() - start

    print("Result:" + str(result))
    print("Duration: "+ str(duration))
    print("End of Testing with Face Recognition lib")

    print("\n\n")


    print("Testing with fotofaces2.0 ....")

    start = time.time()

    result = fotofaces2_0()
    duration = time.time() - start

    print("Result:" + str(result))
    print("Duration: "+ str(duration))
    print("End of Testing with fotofaces2.0")
    


###########     Better refactor do old fotofaces

def testFotoFaces2_0():

    image1 = cv2.imread(sys.argv[1])
    image2 = cv2.imread(sys.argv[2])

    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    shape, bb, raw_shape = fotofaces.detect_face(gray)
    tolerance = fotofaces.face_verification(image1, raw_shape, {}, image2)
    return tolerance <= 0.6


###########     face comparison with Face_recognition lib

def testOldFotofaces():
    imagePath1 = sys.argv[1]
    imagePath2 = sys.argv[2]

    image1 = cv2.imread(imagePath1)
    image2 = cv2.imread(imagePath2)
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    shape, bb, raw_shape = fotofaces.detect_face(gray)
    tolerance = fotofaces.face_verification(image1, raw_shape, {}, image2)
    return tolerance <= 0.6


###########     face comparison with Face_recognition lib

def testFaceRecognitionLib():
    #load
    imagePath1 = sys.argv[1]
    imagePath2 = sys.argv[2]

    image1 = cv2.imread(imagePath1)
    image2 = cv2.imread(imagePath2)


    #comparison
    knownImage = fc.load_image_file(imagePath1)
    unknownImage = fc.load_image_file(imagePath2)


    known_encoding = fc.face_encodings(knownImage)[0]
    unknown_encoding = fc.face_encodings(unknownImage)[0]

    results = fc.compare_faces([known_encoding], unknown_encoding)


    #Draw rectanglens


    for img in [image1, image2]:
        face_locations = fc.face_locations(img)

        for face in face_locations:
            [x, y ,w, h] = dlib_to_shape(face)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        
    #print(results)
    return results[0]
    #cv2.imshow("first image", image1)
    #cv2.imshow("second image", image2)
    #cv2.waitKey(0)

def fotofaces2_0():
    #ficheiro para comparação de caras

    photo = cv2.imread(sys.argv[1])
    reference = cv2.imread(sys.argv[2])
    

    #deteção de uma cara nas fotos
    reference_raw_shape = detect_face(reference)[2]
    photo_raw_shape = detect_face(photo)[2]

    #buscar o chip da cara
    reference_chip = dlib.get_face_chip(reference, reference_raw_shape) 
    photo_chip = dlib.get_face_chip(photo, photo_raw_shape) 

    #usar face recognition file
    reference_descriptor = np.asarray(facerec.compute_face_descriptor(reference_chip)) 
    photo_descriptor = np.asarray(facerec.compute_face_descriptor(photo_chip)) 

    #resultado da comparação se for inferior a 0.6 são a mesma pessoa
    tolerance = np.linalg.norm(reference_descriptor - photo_descriptor)
    return tolerance <= 0.6


    

""" def detect_face(img):
    #ficheiro para deteção de uma cara
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #cena para detetar caras (tbm n sei muito os detalhes)
    
    #conversão para cinzentos
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces_in_img = faceCascade.detectMultiScale(   #algoritmo de detetação
            gray_img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        ) 

    if len(faces_in_img) != 1:
        print("Error: Number of faces found -> " + str(len(faces_in_img)))
        #return None

    [x, y, w, h] = faces_in_img[0] 
    #conversão para dlib rectangle
    return dlib.full_object_detection(rect = dlib.rectangle(left=x, top=y, right=w+x, bottom=h+y)) """

    
def detect_face(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray_img, 1)
    max_area, max_bb, max_shape, raw_shape = (0, None, None, None)
    for (z, rect) in enumerate(rects):
        if rect is not None and rect.top() >= 0 and rect.right() < gray_img.shape[1] and rect.bottom() < gray_img.shape[0] and rect.left() >= 0:
            predicted = predictor(gray_img, rect)
            bb = rect_to_bb(rect)
            area = bb_area(bb)
            # only returns the largest bounding box to avoid smaller false positives
            if area > max_area:
                max_area = area
                max_bb = bb
                max_shape = shape_to_np(predicted)
                raw_shape = predicted
    return max_shape, max_bb, raw_shape

    
    

def face_verification(frame, shape, data, reference):
	candidate = dlib.get_face_chip(frame, shape)   
	candidate_descriptor = facerec.compute_face_descriptor(candidate) 
	# reads the PACO picture
	gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
	shape, bb, raw_shape = detect_face(gray)
	data["Face Reference Detected"] = False
	if bb is not None:
		data["Face Reference Detected"] = True
		old = dlib.get_face_chip(reference, raw_shape)   
		old_descriptor = facerec.compute_face_descriptor(old) 

		candidate_descriptor = np.asarray(candidate_descriptor)
		old_descriptor = np.asarray(old_descriptor)
		dist = np.linalg.norm(old_descriptor - candidate_descriptor)
		return dist


main()

# https://face-recognition.readthedocs.io/en/latest/_modules/face_recognition/api.html#batch_face_locations