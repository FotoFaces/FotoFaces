import face_recognition as fc
import cv2
import sys

imagePath = sys.argv[1]
image = fc.load_image_file(imagePath)
face_locations = fc.face_locations(image)


print(face_locations)
image = cv2.imread(imagePath)

for (x, y, w, h) in face_locations:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)





# https://face-recognition.readthedocs.io/en/latest/_modules/face_recognition/api.html#batch_face_locations