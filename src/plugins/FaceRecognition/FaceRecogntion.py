from logging import Logger

from engine import PluginCore
from model import Meta


class FaceRecogntion(PluginCore):

    def __init__(self, logger: Logger) -> None:
        super().__init__(logger)
        self.meta = Meta(
            name='Face Recognition plugin',
            description='Plugin that returns a distance between two faces within a roi, the bigger the distance the lower the people look a like.Meaning they are not the same person.',
            version='0.0.1'
        )

    def invoke(self, **args):
        """
            Logic of the plugin
            :args is a dictionaire
            :returns a a value related to the metric analysed
        """
        
        self._logger.debug(f'Command: {command} -> {self.meta}')
        #ficheiro para comparação de caras

        photo = cv2.imread(args["new_photo"])
        reference = cv2.imread(args["reference"])
        

        #deteção de uma cara nas fotos
        reference_raw_shape = detect_face(reference)[2]
        photo_raw_shape = detect_face(photo)[2]

        #buscar o chip da cara
        reference_chip = dlib.get_face_chip(reference, reference_raw_shape) 
        photo_chip = dlib.get_face_chip(photo, photo_raw_shape) 

        #usar face recognition file
        reference_descriptor = np.asarray(facerec.compute_face_descriptor(reference_chip)) 
        photo_descriptor = np.asarray(facerec.compute_face_descriptor(photo_chip)) 

        #resultado da comparação se for inferior a 0.6 são a mesma pessoa
        tolerance = np.linalg.norm(reference_descriptor - photo_descriptor)
        return [tolerance <= 0.6]
    
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


    
