import FaceRecognition 
import pytest




def compareGonçaloOldWithNoGlasses():
    args = {"new_photo": "tests/GoncaloNoGlasses.jpg", "reference": "tests/GoncaloOldFoto.jpg"}
    fc = FaceRecognition()
    retval= fc.invoke(args)
    assert len(retval) == 1
    assert retval[0] == True

def compareGonçaloWithPedro():
    args = {"new_photo": "tests/GoncaloOldGlasses.jpg", "reference": "tests/Pedro.jpg"}
    fc = FaceRecognition()
    retval= fc.invoke(args)
    assert len(retval) == 1
    assert retval[0] == False


