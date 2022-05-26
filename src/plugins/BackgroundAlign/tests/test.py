import cv2
import math
import numpy as np
import sys
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def check_the_line(slope, limit):
    if ( 0 < slope <= limit) or (360 - limit <= slope <= 360) or (180 - limit <= slope < 180 + limit):
        return "h"
    elif (90 - limit < slope < 90 + limit) or (270 - limit < slope < 270 + limit):
        return "v"
    return "o"

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

# Detects faces and only returns the largest bounding box
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

def rotate(image, shape):
    dY = shape[36][1] - shape[45][1]
    dX = shape[36][0] - shape[45][0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    dst = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    # transform points
    ones = np.ones(shape=(len(shape), 1))
    points_ones = np.hstack([shape, ones])
    new_shape = M.dot(points_ones.T).T
    new_shape = new_shape.astype(int)

    return dst, new_shape



image = cv2.imread(sys.argv[1])
shape = detect_face(image)[0]

image, shape = rotate(image, shape)

cv2.imshow("rotate", cv2.resize(image, (500, 500)))		#ver a imagem auxiliar com as linhas verticais e horizontais 
cv2.waitKey(0)

image = cv2.GaussianBlur(image, (3, 3), 0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray,219,220)

#HOUGH LINE TRANSFORM OPENCV 

#		---Probabilistic Hough Line---
#img = image 
#minLineLength = 100
#maxLineGap = 1
#lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)

#print("Linhas:")
#print(lines[0])

#for i in range(0,len(lines)):
#	for x1,y1,x2,y2 in lines[i]:
#		cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

#cv2.imshow("Hough Line Transform", img)


#		---"Normal" Hough Line---
#lines = cv2.HoughLines(edges,1,np.pi/180,120)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 75, None, 0, 0)

#print("Linhas:")
#print(len(lines))

if len(lines) > 0:
    n_lines = len(lines)	#numero de linhas detetadas
    n_horizontal = 0
    n_vertical = 0
    n_rest = 0

    aux_image = image

    for i in range(0,len(lines)):
        for rho,theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            #cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
            if y1 != 0 and y2 != 0 and x1 != 0 and x2 != 0 and (y1 - y2) != 0 and (x1 - x2) != 0:
                slope_as_angle = math.atan((y1 - y2) / (x1 - x2))
                slope_as_angle = math.degrees(math.atan2((y1 - y2), (x1 - x2)))

                #print("Slope in degrees: ", slope_as_angle)
                #print("----------------------")

                limit = 10		# dez graus de liberdade para dizer que a linha é horizontal

                if check_the_line(slope_as_angle, limit=limit) == "h":
                    n_horizontal = n_horizontal + 1
                    cv2.line(aux_image,(x1,y1),(x2,y2),(0,255,0),2)		#pintar as linahas horizontais 
                elif check_the_line(slope_as_angle, limit=limit) == "v":
                    n_vertical = n_vertical + 1
                    cv2.line(aux_image,(x1,y1),(x2,y2),(255,0,0),2)		#pintar as linhas verticais 
                else:
                    n_rest = n_rest + 1
                    cv2.line(aux_image,(x1,y1),(x2,y2),(0,0,255),2)		#pintar as linhas verticais 

    print("Horizontal lines number:", n_horizontal)
    print("Vertical lines number:", n_vertical)
    print("Rest lines number:", n_rest)

    cv2.imshow("Hough Line", cv2.resize(aux_image, (500, 500)))		#ver a imagem auxiliar com as linhas verticais e horizontais 
    cv2.waitKey(0)

if (n_vertical > n_rest) or (n_horizontal > n_rest) or (n_horizontal + n_vertical > n_rest): 
    print("Good Background")
else:
    print("Tilted Background")

