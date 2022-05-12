import sys
import numpy as np
import cv2
import dlib

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
    return max_shape

def dist_ratio(p1, p2, center):
    p1_dist = abs(center-p1)
    p2_dist = abs(center-p2)

    if p1_dist > p2_dist:
        ratio = (p2_dist/p1_dist)*100
    else:
        ratio = (p1_dist/p2_dist)*100

    return ratio

path = sys.argv[1]
image = cv2.imread(path)
shape = detect_face(image)





ratio = []
kernel = np.ones((3, 3), np.uint8)
# Left eye
x1 = shape[37][0]
y1 = int((shape[37][1]+shape[36][1])/2)
x2 = int((shape[39][0]+shape[40][0])/2)
y2 = int((shape[39][1]+shape[40][1])/2)

tl1 = (min(x1, x2), min(y1, y2))
br1 = (max(x1, x2), max(y1, y2))

if tl1[1] != br1[1] and tl1[0] != br1[0]:
    e1 = image[tl1[1]:br1[1],tl1[0]:br1[0]]
    e1p = cv2.cvtColor(e1, cv2.COLOR_BGR2GRAY)
    e1p = cv2.bilateralFilter(e1p, 10, 30, 30)
    e1p = cv2.erode(e1p, kernel, iterations=3)
    ret, e1p = cv2.threshold(e1p,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(e1p, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        contours = max(contours, key = cv2.contourArea)
        try:
            moments = cv2.moments(contours)
            x = int(moments["m10"] / moments["m00"])
            y = int(moments["m01"] / moments["m00"])
            e1_og = (tl1[0]+x, tl1[1]+y)
            ratio1 = dist_ratio(shape[36][0], shape[39][0], e1_og[0])
            ratio.append(ratio1)
        except (IndexError, ZeroDivisionError):
            pass

# Right eye
x1 = shape[44][0]
y1 = int((shape[44][1]+shape[45][1])/2)
x2 = int((shape[42][0]+shape[47][0])/2)
y2 = int((shape[42][1]+shape[47][1])/2)

tl2 = (min(x1, x2), min(y1, y2))
br2 = (max(x1, x2), max(y1, y2))

if tl2[1] != br2[1] and tl2[0] != br2[0]:
    e2 = image[tl2[1]:br2[1],tl2[0]:br2[0]]

    e2p = cv2.cvtColor(e2, cv2.COLOR_BGR2GRAY)
    e2p = cv2.bilateralFilter(e2p, 10, 30, 30)
    e2p = cv2.erode(e2p, kernel, iterations=3)
    ret, e2p = cv2.threshold(e2p,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(e2p, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        contours = max(contours, key = cv2.contourArea)
        try:
            moments = cv2.moments(contours)
            x = int(moments["m10"] / moments["m00"])
            y = int(moments["m01"] / moments["m00"])
            e2_og = (tl2[0]+x, tl2[1]+y)
            ratio2 = dist_ratio(shape[42][0], shape[45][0], e2_og[0])
            ratio.append(ratio2)
        except (IndexError, ZeroDivisionError):
            pass

focus = 0
if len(ratio) == 2:
    focus = sum(ratio)/len(ratio)
elif len(ratio) == 1:
    focus = ratio[0]

print(focus)