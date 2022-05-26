

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

	gray = candidate
	shape_0, bb, raw_shape = detect_face(gray)

	background = -1
		
	if shape_0 is not None:	
		if bb is None:
			print("No face Detected!")
		else:
			print("Face Detected!")

		final_img, final_shape = rotate(candidate, shape_0)	#Face Alignment 

		if final_shape is not None:

			#cv2.imshow("Paco Image", final_img)

			print("Image Face Shape: ", final_shape)

			inp = cv2.dnn.blobFromImage(final_img, scalefactor=1.0, size=(args.width, args.height),
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

			background = background_rotation(final_img, final_shape, out)

	if background == 1:
		return True
	elif background == 0:
		return False
	else:
		return False