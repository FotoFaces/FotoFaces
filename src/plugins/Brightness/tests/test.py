import sys
import numpy as np
import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../../../shape_predictor_68_face_landmarks.dat")

CROP_ALPHA = 0.60

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

def cropping(image, shape):
	aux = shape[0] - shape[16]
	distance = np.linalg.norm(aux)

	h = int(distance)

	middle_X = int((shape[0][0] + shape[16][0])/2)
	middle_Y = int((shape[19][1] + shape[33][1])/2)

	x1 = int(middle_X-h*CROP_ALPHA)
	y1 = int(middle_Y-h*CROP_ALPHA)
	x2 = int(middle_X+h*CROP_ALPHA)
	y2 = int(middle_Y+h*CROP_ALPHA)
	tl = (x1, y1)
	br = (x2, y2)

	if x1 >= 0 and y1 >= 0 and x2 < image.shape[1] and y2 < image.shape[0]:
		roi = image[tl[1]:br[1],tl[0]:br[0]]
		return roi
	else:
		return None


def avg_bright(roi, shape):

    image = cropping(roi, shape)

    #height, width = image.shape[:2]
    #res = cv2.resize(image,(width//4, height//4), interpolation = cv2.INTER_AREA)
    #cv2.imshow("face", res)
    #cv2.waitKey(0)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    avg = np.mean(v)
    print(avg)
    return avg > 90




def func(path_img, expect):
    image = cv2.imread(path_img)
    shape = detect_face(image)[0]
    
    return avg_bright(image,shape) == expect

def test_bright_1():
    assert func( "images/bright_vicente_1.jpg", True)
def test_bright_2():
    assert func( "images/bright_vicente_2.jpg", True)
def test_bright_3():
    assert func( "images/bright_vicente_3.jpg", True)
def test_bright_4():
    assert func( "images/bright_vicente_4.jpg", True)
def test_bright_5():
    assert func( "images/bright_vicente_5.jpg", True)
def test_bright_6():
    assert func( "images/bright_vicente_6.jpg", False)
def test_FaceBright_BackgroundDark_1():
    assert func( "images/bright_Pedro_1.jpg", True)
def test_FaceBright_BackgroundDark_2():
    assert func( "images/bright_Pedro_2.jpg", True)




 