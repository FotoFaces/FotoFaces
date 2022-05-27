import sys
import numpy as np
import cv2
import dlib
import pytest

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../../../shape_predictor_68_face_landmarks.dat")


def detect_sunglasses(image,shape):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Left eye
    if shape[40][0] != shape[41][0]:
        x_coords = (shape[40][0], shape[41][0])
        left_point = x_coords.index(min(x_coords))

        aux = shape[41] - shape[40]
        distance = np.linalg.norm(aux)
        h = int(distance)

        tl1 = (shape[40+left_point][0], shape[40+left_point][1])
        br1 = (shape[40+left_point][0]+h, shape[40+left_point][1]+h)
    else:
        tl1 = (shape[41][0], shape[41][1])
        br1 = (shape[41][0]+1, shape[41][1]+1)

    # Right eye
    if shape[46][0] != shape[47][0]:
        x_coords = (shape[46][0], shape[47][0])
        left_point = x_coords.index(min(x_coords))

        aux = shape[47] - shape[46]
        distance = np.linalg.norm(aux)
        h = int(distance)

        tl2 = (shape[46+left_point][0], shape[46+left_point][1])
        br2 = (shape[46+left_point][0]+h, shape[46+left_point][1]+h)
    else:
        tl2 = (shape[47][0], shape[47][1])
        br2 = (shape[47][0]+1, shape[47][1]+1)


    e1 = hsv_image[tl1[1]:br1[1],tl1[0]:br1[0]]

    e2 = hsv_image[tl2[1]:br2[1],tl2[0]:br2[0]]

    skin_reference = hsv_image[shape[30][1], shape[30][0]]
    h1, s1, v1 = cv2.split(e1)
    h2, s2, v2 = cv2.split(e2)

    S_average = (np.mean(s1)+np.mean(s2))/2
    V_average = (np.mean(v1)+np.mean(v2))/2

    S_diff = abs(skin_reference[1] - S_average)
    V_diff = abs(skin_reference[2] - V_average)
    #print("S_diff",S_diff)
    #print("V_diff",V_diff)
    if S_diff < 90 and V_diff < 90:
        return False
    else :
        return True

def bb_area( bb):
        return (bb[0] + bb[2]) * (bb[1] + bb[3])

# Converts dlib format to numpy format
def shape_to_np( shape):
    landmarks = np.zeros((68, 2), dtype=int)

    for i in range(0, 68):
        landmarks[i] = (shape.part(i).x, shape.part(i).y)

    return landmarks

# Converts dlib format to opencv format
def rect_to_bb( rect):
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


#image = cv2.imread(sys.argv[1])
#shape = detect_face(image)[0]

#print(detect_sunglasses(image,shape))


def func(path_img, expect):
    image = cv2.imread(path_img)
    shape = detect_face(image)[0]

    return detect_sunglasses(image,shape) == expect

def test_person():
    assert func( "images/bright_mini_1.jpg", False)
def test_person_with_glasses():
    assert func("images/bright_vicente_1.jpg",False)
def test_person_with_Sunglasses_no_visible_eyes():
    assert func("images/sunglasses_1.jpg",True)
def test_person_light_yellow_Sunglasses_visible_eyes():
    assert func("images/sunglasses_2.jpg",True)
def test_black_person_Sunglasses_slight_visible_eyes():
    assert func("images/sunglasses_3.jpg",True)
def test_person_Pink_Sunglasses_slight_visible_eyes():
    assert func("images/sunglasses_4.jpg",True)
def test_person_Red_Sunglasses_slight_visible_eyes():
    assert func("images/sunglasses_5.jpg",True)
def test_person_yellow_Sunglasses_no_visible_eyes_with_reflexition():
    assert func("images/sunglasses_6.jpg",True)
def test_person_Sunglasses_visible_eyes():
    assert func("images/sunglasses_7.jpg",True)