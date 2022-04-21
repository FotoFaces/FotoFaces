import face_recognition as fc
import cv2
import sys



def rect_to_bb(face):
	x = face[3]
	y = face[0]
	w = face[1] - x
	h = face[2] - y
	return [x, y, w, h]





###########     face comparison

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
            [x, y ,w, h] = rect_to_bb(face)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        
    print(results)
    cv2.imshow("first image", image1)
    cv2.imshow("second image", image2)
    cv2.waitKey(0)

# https://face-recognition.readthedocs.io/en/latest/_modules/face_recognition/api.html#batch_face_locations