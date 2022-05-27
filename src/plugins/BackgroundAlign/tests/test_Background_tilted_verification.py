import cv2
import math
import numpy as np
import sys
import dlib

net = cv2.dnn.readNet("../deploy.prototxt", "../hed_pretrained_bsds.caffemodel")
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

""" def rotate(image, shape):
	dY = shape[36][1] - shape[45][1]
	dX = shape[36][0] - shape[45][0]
	angle = np.degrees(np.arctan2(dY, dX)) - 180

	rows,cols = image.shape[:2]
	M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
	dst = cv2.warpAffine(image,M,(cols,rows), borderMode=cv2.BORDER_REFLECT)

	#transform points
	ones = np.ones(shape=(len(shape), 1))
	points_ones = np.hstack([shape, ones])
	new_shape  = M.dot(points_ones.T).T
	new_shape = new_shape.astype(int)

	return dst, new_shape """

def find_lines_using_hough_lines_P(img_edges):
    
	img_edges = cv2.cvtColor(img_edges, cv2.COLOR_BGR2GRAY)
	lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=130, maxLineGap=5)

	return lines

def calculate_angle_P(lines, img_before):
	angles = []
	#print('Lines:', lines)
	for i in range(0,len(lines)):
		for x1,y1,x2,y2 in lines[i]:
   
			cv2.line(img_before,(x1,y1),(x2,y2),(0,255,0),2)
			angle = math.degrees(math.atan2(y2 - y1, x2 - x1))  # find angle of line connecting (0,0) to (x,y) from +ve x axis
	#		print('Tilt angle for x1, y1, x2, y2 {} is {}'.format([x1, y1, x2, y2], angle) )
			angles.append(angle)

	median_angle = np.median(angles)
	#print('median_angle:', median_angle)

	return median_angle

def rotate_degree(image, degree):
	
	angle = degree

	rows,cols = image.shape[:2]
	M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
	dst = cv2.warpAffine(image,M,(cols,rows), borderMode=cv2.BORDER_REFLECT)

	return dst


def foresty_lines(img_edges):
	print("foresty_lines")
	#global angle_array

	angle = 1000

	Bigsum = 0
	forangle = 0

	#Usaar a altura da imagem para definir o treshold da linha, neste caso é considerado uma linha se tiver mais de 1/5 dos pixeis do tamanho da altura da imagem
	img_height = img_edges.shape[0]

	threshold = img_height/5
	#threshold = 100 # se a linhar tiver mais de x pixeis então é aceite

	for angle in range(-90, 90):
		#print("angle", angle)
		#chamar a funcao de rotate já implementada mas
		new_img = rotate_degree(img_edges, angle)

		img_edges_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

		(thresh, im_bw) = cv2.threshold(img_edges_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

		bestlinesum = 0

		line = 0    #numero de pontos de uma linha 
		limit = 10  #numero de pixeis de espaçamento para se considerar uma linha 
		#print("im_bw", shape[1])
		#print("im_bw", shape[0])
		for column in range (0, im_bw.shape[1]):

			exit = limit 

			for row in range(0,im_bw.shape[0]):

				pixel = im_bw[row, column]

				if pixel == 0:   #black pixel 
					exit = exit -1 
					line = line -1 
				elif pixel == 255 :
					line = line + 1 
					exit = limit 

				if exit <= 0:
					if line > threshold:
						bestlinesum = bestlinesum + line
                    
					line = 0
                
				if row == (im_bw.shape[0] -1) :
					if line > threshold:
						bestlinesum = bestlinesum + line 

		if bestlinesum > Bigsum:
			Bigsum = bestlinesum
			forangle = angle 

	return forangle



def background_rotation(image, shape, img_edges):

	#global x 

	back = -1

	#Excluir a região da face da imagem para ficar apenas com o background 
	AUX = 0.95

	aux = shape[0] - shape[16]
	distance = np.linalg.norm(aux)
	h = int(distance)

	middle_X = int((shape[0][0] + shape[16][0])/2)
	middle_Y = int((shape[19][1] + shape[29][1])/2)

	x1 = int((shape[0][0])) - int(0.2*h)
	y1 = int(middle_Y-h*AUX)
	x2 = int((shape[16][0])) + int(0.2*h)
	y2 = int((shape[57][1]))

	#Excluir a região abaixo da linha da boca 
	x1_1 = int(0)
	y1_1 = int((shape[57][1]))
	x2_1 = int(image.shape[1])
	y2_1 = int(image.shape[0])
	
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #convert BGR to RGB
	#display_image(image)
	image_blur = cv2.GaussianBlur(image, (5, 5), 0)

	gray = cv2.cvtColor(image_blur, cv2.COLOR_RGB2GRAY)

	#cv2.rectangle(img_edges, (x1,y1), (x2,y2), (0, 0, 0), -1)
	#cv2.rectangle(img_edges, (x1_1,y1_1), (x2_1,y2_1), (0, 0, 0), -1)
	#cv2.imshow("img_edges", img_edges)
	#cv2.waitKey(0)
	img_edges_new = img_edges
	lines_P = find_lines_using_hough_lines_P(img_edges_new)
	#print(lines_P)
	if lines_P is not None:
		img_before = image
		#median_angle = calculate_angle(lines, img_before)
		median_angle_P = round(calculate_angle_P(lines_P, img_before))
		#print("Median Angle: ", median_angle)
		print("Median Angle Prob: ", median_angle_P)

		if -5 <= median_angle_P <= 5 or 85 <= median_angle_P <= 90 or -90 <= median_angle_P <= -85:
			back = 1	#Background direito se a média dos ângulos estiver entre estes valores 
		else:
			back = 0
		
	else:
		back = 1	#Background direito se não forem detetadas linhas

	#print(back)

	#img_angle = foresty_lines(img_edges)

	#if -5 <= img_angle <= 5 or 85 <= img_angle <= 90 or -90 <= img_angle <= -85:
	#	back = 1	#Background direito se a média dos ângulos estiver entre estes valores 
	#else:
	#	back = 0

	#cv2.imshow("end", img_edges_new)
	#cv2.waitKey(0)

	return back 


def background_tilt(candidate):
	img_width = candidate.shape[1]
	img_height = candidate.shape[0]

	if img_width <= 1000 or  img_height <= 1000:
		scale_factor = 1
	if 1000 < img_width <= 2000 or 1000 < img_height <= 2000:
		scale_factor = 2
	if img_width > 2000 or img_height > 2000:
		scale_factor = 3

	candidate = cv2.resize(candidate, (int(img_width/scale_factor),int(img_height/scale_factor)))

	shape_0, bb, raw_shape = detect_face(candidate)

	background = -1
		
	if shape_0 is not None:	
		if bb is None:
			print("No face Detected!")
		else:
			print("Face Detected!")

		#final_img, final_shape = rotate(candidate, shape_0)	#Face Alignment 
		final_img = candidate
		final_shape = shape_0

		if final_shape is not None:

			#cv2.imshow("Paco Image", final_img)

			#print("Image Face Shape: ", final_shape)

			inp = cv2.dnn.blobFromImage(final_img, scalefactor=1.0, size=(500, 500),
							mean=(104.00698793, 116.66876762, 122.67891434),
							swapRB=False, crop=False)
			
			net.setInput(inp)
			out = net.forward()
			out = out[0, 0]
			out = cv2.resize(out, (final_img.shape[1], final_img.shape[0]))
			out = 255 * out
			out = out.astype(np.uint8)
			out=cv2.cvtColor(out,cv2.COLOR_GRAY2BGR)
			con=np.concatenate((final_img,out),axis=1)
			#cv2.imshow("final_img", final_img)
			#cv2.waitKey(0)
			#cv2.imshow("out", out)
			#cv2.waitKey(0)
			
			#cv2.destroyAllWindows()

			background = background_rotation(final_img, final_shape, out)

	if background == 1:
		return True
	elif background == 0:
		return False
	else:
		return False




def func(path_img, expect):
    image = cv2.imread(path_img)
    return background_tilt(image) == expect

def test_background_cursed():
    assert func( "images/Background_cursed1.png", False)
def test_background_cursed2():
     assert func("images/Background_cursed2.png",False)
def test_background_slightly_cursed():
     assert func("images/Background_1.png",False)
def test_background_very_slightly():
     assert func("images/Background_2.png",False)
def test_img_no_rotated3():
     assert func("images/Open_eyes_3.jpg",True)


#test_img_no_rotated3()
test_background_slightly_cursed()