from flask import request, Flask
import numpy as np
import cv2
import math
import dlib
import base64
import json

app = Flask(__name__)

""" 
Reads two input images and an identifier: 
- Candidate picture 
- PACO reference picture

Outputs a dictionary with the cropped image (depending on some requirements), metrics and the same identifier.
"""

# Cropping threshold (for higher values the cropping might be bigger than the image itself
# which will make the app consider that the face is out of bounds)
CROP_ALPHA = 0.95

# loads everything needed
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../../../shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("../../../dlib_face_recognition_resnet_model_v1.dat")

# Calculates the bounding box area
def bb_area(bb):
	return (bb[0]+bb[2])*(bb[1]+bb[3])


# Converts dlib format to numpy format
def shape_to_np(shape):
	landmarks = np.zeros((68,2), dtype = int)

	for i in range(0,68):
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
def detect_face(gray_image):
	rects = detector(gray_image, 1)
	max_area, max_bb, max_shape, raw_shape = (0, None, None, None)

	for (z, rect) in enumerate(rects):
		if rect is not None and rect.top() >= 0 and rect.right() < gray_image.shape[1] and rect.bottom() < gray_image.shape[0] and rect.left() >= 0:
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


# Applies rotation correction
def rotate(image, shape):
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

	return dst, new_shape


# Crops the image into the final PACO image
def cropping(image, shape, data):
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

	data["Crop Position"] = (x1,y1,x2,y2)
	if x1 >= 0 and y1 >= 0 and x2 < image.shape[1] and y2 < image.shape[0]:
		roi = image[tl[1]:br[1],tl[0]:br[0]]
		return roi
	else:
		return None

# Detects blinks using the Eye Aspect Ratio formula and returns True if eyes open
def eyes_open(shape):
	leftEAR = (np.linalg.norm(shape[37]-shape[41]) + np.linalg.norm(shape[38]-shape[40]))/(2*np.linalg.norm(shape[36]-shape[39]))
	rightEAR = (np.linalg.norm(shape[43]-shape[47]) + np.linalg.norm(shape[44]-shape[46]))/(2*np.linalg.norm(shape[42]-shape[45]))

	avg = (leftEAR+rightEAR)/2
	return avg


# Detects sunglasses by comparing the region below the eyes with the skin color (HSV) and returns True if no sunglasses
def no_sunglasses(image, shape):
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


	return (S_diff, V_diff)


# estimates head pose by transforming the 2D facial landmarks into 3D world coordinates and calculating the Euler angles
def head_pose(im, features):
	size = im.shape

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
	pitch = math.degrees(math.asin(math.sin(pitch)))
	roll = -math.degrees(math.asin(math.sin(roll)))
	yaw = math.degrees(math.asin(math.sin(yaw)))

	return (pitch, roll, yaw)


# Calculates distance ratio between the pupil and eye corners
def dist_ratio(p1, p2, center):
	p1_dist = abs(center-p1)
	p2_dist = abs(center-p2)

	if p1_dist > p2_dist:
		ratio = (p2_dist/p1_dist)*100
	else:
		ratio = (p1_dist/p2_dist)*100

	return ratio


# Estimates gaze by detecting the pupils through morphological operations, finding their contours and estimating their distance to the corners
def gaze(image, shape):
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
	
	return focus


# Computes image quality (0 being the best quality and 100 being the worst)
def image_quality(roi):
	img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	score = cv2.quality.QualityBRISQUE_compute(img, "brisque_model_live.yml", "brisque_range_live.yml") 
	obj = cv2.quality.QualityBRISQUE_create("brisque_model_live.yml", "brisque_range_live.yml")
	score = obj.compute(img)[0]
	return score


# Does face verification using the dlib trained model
def face_verification(frame, shape, data, reference):
	candidate = dlib.get_face_chip(frame, shape)   
	candidate_descriptor = facerec.compute_face_descriptor(candidate) 
	# reads the PACO picture
	gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
	shape, bb, raw_shape = detect_face(gray)
	data["Face Reference Detected"] = False
	if bb is not None:
		data["Face Reference Detected"] = True
		old = dlib.get_face_chip(reference, raw_shape)   
		old_descriptor = facerec.compute_face_descriptor(old) 

		candidate_descriptor = np.asarray(candidate_descriptor)
		old_descriptor = np.asarray(old_descriptor)
		dist = np.linalg.norm(old_descriptor - candidate_descriptor)
		return dist


# Calculates the image brightness 
def average_brightness(roi):
	hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(hsv)
	avg = np.mean(v)
	return avg


# Checks if the image is gray
def is_gray(img):
	b,g,r = cv2.split(img)
	if np.array_equal(b,g) and np.array_equal(b,r):
		return False
	return True

@app.route("/", methods=["POST"])
def upload_image():
	if "candidate" in request.form.keys() and "id" in request.form.keys():
		img1 = request.form["candidate"]
		identifier = request.form['id']
		identifier_decoded = identifier
		candidate = cv2.imdecode(np.frombuffer(base64.b64decode(img1), np.uint8), cv2.IMREAD_COLOR)

		data = {}
		data["Colored Picture"] = is_gray(candidate)
		if data["Colored Picture"] == False:
			dict_data = {'id':identifier_decoded, 
						'feedback':json.dumps(data)}
			return dict_data
		else:
			# reads the candidate picture
			gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
			shape, bb, raw_shape = detect_face(gray)
			data["Face Candidate Detected"] = True
			if bb is None:
				# No face detected
				data["Face Candidate Detected"] = False
				dict_data = {'id':identifier_decoded, 
							'feedback':json.dumps(data)}
				return dict_data
			else:
				image, shape = rotate(candidate, shape)
				roi = cropping(image, shape, data) 
				data["Cropping"] = True
				if roi is None:
					# Face is not centered and/or too close to the camera
					data["Cropping"] = False
					dict_data = {'id':identifier_decoded, 
								'feedback':json.dumps(data)}
					return dict_data
				else:
				# The face was detected and is centered
					data["Resize"] = 500/roi.shape[0]
					final_img = cv2.resize(roi, (500,500))

					# Calculates the required parameters to validate the picture
					if "reference" in request.form.keys():
						img2 = request.form["reference"]
						reference = cv2.imdecode(np.frombuffer(base64.b64decode(img2), np.uint8), cv2.IMREAD_COLOR)
						data["Face Recognized"] = face_verification(candidate, raw_shape, data, reference)
					data["Frontal Face"] = head_pose(image, shape)
					data["No Sunglasses"] = no_sunglasses(image, shape)
					data["Eyes Open"] = eyes_open(shape)
					data["Gaze"] = gaze(image, shape)
					data["Brightness"] = average_brightness(candidate)
					data["Image Quality"] = image_quality(final_img)

					_, img_encoded = cv2.imencode(".jpg", final_img)
					
					dict_data = {'id':identifier_decoded, 
								'feedback':json.dumps(data), 'cropped':base64.b64encode(img_encoded).decode('ascii')}

					return dict_data

	return ("", 204)

if __name__ == "__main__":
	app.run()
