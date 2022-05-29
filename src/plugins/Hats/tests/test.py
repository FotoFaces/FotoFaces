import cv2
import math
import numpy as np
import sys
import dlib
import keras
from keras.models import model_from_json
import tensorflow as tf

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


# Detects sunglasses using  Machine Learning trained model
def detect_hats(image, shape):
	
	roi_cropp = cropping_hats(image, shape)
	#print("roi_cropp",roi_cropp)
	img_size = 150
	resized_arr = cv2.resize(roi_cropp, (img_size, img_size)) # Reshaping images to preferred size
	resized_arr = cv2.cvtColor(resized_arr, cv2.COLOR_RGB2BGR)

	x_test = []
	x_test.append(resized_arr)
	x_test = np.array(x_test)
	x_test = (x_test / 127.5 ) - 1

	# load json and create model
	json_file = open('../model_tf_hats_head_cropp_hand_final.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("../model_tf_hats_head_cropp_hand_final.h5")
	print("Loaded model from disk")

	# evaluate loaded model on test data
	loaded_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # Very low learning rate
				loss=keras.losses.BinaryCrossentropy(from_logits=True),
				metrics=[keras.metrics.BinaryAccuracy()])

	predictions = (loaded_model.predict(np.array(x_test)) > 0.5).astype("int32")

	#print("Prediction Hats: ", predictions[0])

	if predictions[0] == 1:
		return True

	return False


# Crops the image into the head reagion to give as input to the hats trained model 
def cropping_hats(image, shape):

    CROP_ALPHA = 0.75

    aux = shape[0] - shape[16]
    distance = np.linalg.norm(aux)
    h = int(distance)

    middle_X = int((shape[0][0] + shape[16][0])/2)
    middle_Y = int((shape[19][1] + shape[29][1])/2)

    x1 = int((shape[0][0]))
    y1 = int(middle_Y-h*CROP_ALPHA)
    x2 = int((shape[16][0]))
    y2 = int((shape[19][1]))
    tl = (x1, y1)
    br = (x2, y2)

    if x1 >= 0 and y1 >= 0 and x2 < image.shape[1] and y2 < image.shape[0]:
        roi = image[tl[1]:br[1],tl[0]:br[0]]
        return roi

    else: 
        return None


def func(path_img, expect):
    image = cv2.imread(path_img)
    shape = detect_face(image)[0]
    
    return detect_hats(image,shape) == expect

def test_person_with_no_hat():
    assert func( "images/hat_person_no.jpg", False)

def test_person_with_campaign_hat():
    assert func( "images/hat_person_1.jpg", True)
def test_person_with_hat_showing_forehead():
    assert func("images/hat_person_2.jpg",True)
def test_person_with_bobble_hat():
    assert func("images/hat_person_3.jpg",True)
def test_person_with_large_chupalla_hat():
    assert func("images/hat_person_4.jpg",True)
def test_person_with_baseball_hat():
    assert func("images/hat_person_5.jpg",True)
def test_person_samll_straw_hat():
    assert func("images/hat_person_5.jpg",True)

