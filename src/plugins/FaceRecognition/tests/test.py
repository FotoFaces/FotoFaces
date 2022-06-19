import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../../../shape_predictor_68_face_landmarks.dat")

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

def detect_face(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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


path = "images/GoncaloNoGlasses.jpg"
image = cv2.imread(path)
bb = detect_face(image)[1]
height, width = image.shape[:2]
cv2.rectangle(image, (bb[0],bb[1]), (bb[0]+bb[2],bb[1]+bb[3]) , (0,255,0), 3 )
res = cv2.resize(image,(width//4, height//4), interpolation = cv2.INTER_AREA)

cv2.imshow("img",res)
cv2.waitKey(0)
