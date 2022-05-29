
import numpy as np
import cv2
import dlib


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../../../shape_predictor_68_face_landmarks.dat")


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


def has_glasses(image,shape):
    nose_bridge_x = []
    nose_bridge_y = []
    for i in [28,29,30,31,33,34,35]:
            nose_bridge_x.append(shape[i][0])
            nose_bridge_y.append(shape[i][1])

    x_min = min(nose_bridge_x)
    x_max = max(nose_bridge_x)
    y_min = shape[20][1]
    y_max = shape[31][1]
    img2 = image[y_min:y_max, x_min:x_max]

    img_blur = cv2.GaussianBlur(np.array(img2),(3,3), sigmaX=0, sigmaY=0)
    edges = cv2.Canny(image =img_blur, threshold1=100, threshold2=200)
    edges_center = edges.T[(int(len(edges.T)/2))]

    if 255 in edges_center:
        return True
    return False


def func(path_img, expect):
    image = cv2.imread(path_img)
    shape = detect_face(image)[0]

    return has_glasses(image,shape) == expect

def test_Me_with_glasses():
    assert func( "images/Glasses_1.jpg", True)
def test_person_with_glasses():
    assert func("images/Glasses_2.jpg",True)
def test_person_with__big_glasses():
    assert func("images/Glasses_3.jpg",True)
def test_person_with_small_glasses():
    assert func("images/Glasses_4.jpg",True)
def test_person_with_HUJE_glasses():
    assert func("images/Glasses_5.jpg",True)
def test_person_with_very_big_glasses():
    assert func("images/Glasses_6.jpg",True)
def test_person_with_bottle_coke_glasses():
    assert func("images/Glasses_7.jpg",True)
def test_person_with_no_glasses():
    assert func("images/Glasses_8.jpg",False)
