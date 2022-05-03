from logging import Logger
from engine import PluginCore
from model import Meta


class FaceRecognition(PluginCore):

    def __init__(self, logger: Logger, appCore) -> None:
        super().__init__(logger, appCore)
        self.meta = Meta(
            name='Face Recognition plugin',
            description='Plugin that returns a distance between two faces within a roi, the bigger the distance the lower the people look a like.Meaning they are not the same person.',
            version='0.0.1'
        )
        self.facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.appCore = appCore

    def invoke(self, **args):
        
        """
            Logic of the plugin
            :args is a dictionaire
            :returns a a value related to the metric analysed
        """
        
        self._logger.debug(f'Command: {command} -> {self.meta}')
        #ficheiro para comparação de caras

        photo = cv2.imread(args.get("new_photo"))
        reference = cv2.imread(args.get("reference"))
        

        #deteção de uma cara nas fotos
        reference_raw_shape = detect_face(reference)[2]
        photo_raw_shape = detect_face(photo)[2]

        #buscar o chip da cara
        reference_chip = dlib.get_face_chip(reference, reference_raw_shape) 
        photo_chip = dlib.get_face_chip(photo, photo_raw_shape) 

        #usar face recognition file
        reference_descriptor = np.asarray(self.facerec.compute_face_descriptor(reference_chip)) 
        photo_descriptor = np.asarray(self.facerec.compute_face_descriptor(photo_chip)) 

        #resultado da comparação se for inferior a 0.6 são a mesma pessoa
        tolerance = np.linalg.norm(reference_descriptor - photo_descriptor)
        return [tolerance <= 0.6]
    
    def detect_face(img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray_img, 1)
        max_area, max_bb, max_shape, raw_shape = (0, None, None, None)
        for (z, rect) in enumerate(rects):
            if rect is not None and rect.top() >= 0 and rect.right() < gray_img.shape[1] and rect.bottom() < gray_img.shape[0] and rect.left() >= 0:
                predicted = self.predictor(gray_img, rect)
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


    
