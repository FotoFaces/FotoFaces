import face_recognition as fc
import cv2
import sys
import fotofaces
import time



def rect_to_bb(face):
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
            [x, y ,w, h] = rect_to_bb(face)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        
    #print(results)
    return results[0]
    #cv2.imshow("first image", image1)
    #cv2.imshow("second image", image2)
    #cv2.waitKey(0)


main()

# https://face-recognition.readthedocs.io/en/latest/_modules/face_recognition/api.html#batch_face_locations