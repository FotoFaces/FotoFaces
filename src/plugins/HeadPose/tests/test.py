
import numpy as np
import cv2
import dlib
import math
import pytest


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


def head_pose(image,features):
    if features is None:
        return False
    size = image.shape

    image_points = np.array([
                            features[30],     # Nose
                            features[36],     # Left eye
                            features[45],     # Right eye
                            features[48],     # Left Mouth corner
                            features[54],     # Right mouth corner
                            features[8]		  # Chin
                            ], dtype="double")

    # 3D model points.
    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose
                            (-165.0, 170.0, -135.0),     # Left eye
                            (165.0, 170.0, -135.0),      # Right eye
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0),     # Right mouth corner
                            (0.0, -330.0, -65.0)		 # Chin
                            ])

    # Camera internals
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    rv_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rv_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

    pitch, yaw, roll = [math.radians(x) for x in eulerAngles]
    pitch = abs(math.degrees(math.asin(math.sin(pitch))))
    roll = abs(math.degrees(math.asin(math.sin(roll))))
    yaw = abs(math.degrees(math.asin(math.sin(yaw))))
    print("pitch", pitch)
    print("roll", roll)
    print("yaw", yaw)
    return pitch < 15 and roll < 15 and yaw < 15


def func(path_img, expect):
    image = cv2.imread(path_img)
    shape = detect_face(image)[0]

    return head_pose(image,shape) == expect

def test_face_forward():
    assert func( "images/head_pose1.jpg", True)
def test_face_right():
    assert func("images/head_pose2.jpg",False)
def test_face_UpRight():
    assert func("images/head_pose3.jpg",False)
def test_face_Up():
    assert func("images/head_pose4.jpg",False)
def test_face_UpLeft():
    assert func("images/head_pose5.jpg",False)
def test_face_Left():
    assert func("images/head_pose6.jpg",False)
def test_face_DownLeft():
    assert func("images/head_pose7.jpg",False)
def test_face_Down():
    assert func("images/head_pose8.jpg",False)
def test_face_DownRight():
    assert func("images/head_pose9.jpg",False)

test_face_forward()