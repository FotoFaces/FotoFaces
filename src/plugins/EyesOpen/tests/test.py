import sys
import cv2
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../../../shape_predictor_68_face_landmarks.dat")
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


def detect_face( image):
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





def detect_eyes(shape):
    leftEAR = (
        np.linalg.norm(shape[37] - shape[41])
        + np.linalg.norm(shape[38] - shape[40])
    ) / (2 * np.linalg.norm(shape[36] - shape[39]))
    rightEAR = (
        np.linalg.norm(shape[43] - shape[47])
        + np.linalg.norm(shape[44] - shape[46])
    ) / (2 * np.linalg.norm(shape[42] - shape[45]))

    avg = (leftEAR + rightEAR) / 2

    print(avg)
    return avg > 0.21


def func(path_img, expect):
    image = cv2.imread(path_img)
    shape = detect_face(image)[0]
    
    return detect_eyes(shape) == expect

def test_eyes_open_1():
    assert func( "images/Open_eyes_2.jpg", True)
def test_eyes_open_2():
    assert func( "images/Open_eyes_3.jpg", True)
def test_eyes_open_3():
    assert func( "images/Open_eyes_4.jpg", True)
def test_eyes_open_4():
    assert func( "images/Open_eyes_5.jpg", True)
def test_eyes_open_5():
    assert func( "images/Open_eyes_6.jpg", False)
def test_eyes_open_6():
    assert func( "images/Open_eyes_7.jpg", False)


def test_asian():
    assert func( "images/simu_liu.jpg", True)
def test_asian_2():
    assert func( "images/asian_2.jpg", False)