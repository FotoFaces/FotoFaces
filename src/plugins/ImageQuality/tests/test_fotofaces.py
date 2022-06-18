import cv2
import sys


def img_qua(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.quality.QualityBRISQUE_compute(gray, "../../../brisque_model_live.yml", "../../../brisque_range_live.yml")[0]
    print(score)
    return round(score) < 25


def func(path_img, expect):
    
    return img_qua(path_img) == expect

def test_bad_quality_1():
    assert func( "images/bad_quality.jpg", False)
def test_bad_quality_2():
    assert func( "images/bad_quality2.jpg", False)
def test_bad_quality_3():
    assert func( "images/GoncaloOldFoto.jpg", False)
def test_bad_quality_4():
    assert func( "images/mini_bad.jpg", False)

def test_good_quality_1():
    assert func( "images/neves.jpg", True)
def test_good_quality_2():
    assert func( "images/Pedro.jpg", True)
def test_good_quality_3():
    assert func( "images/vicente_no_blur.jpg", True)
def test_good_quality_4():
    assert func( "images/vicente.png", True)
def test_good_quality_5():
    assert func( "images/vieira.jpeg", True)


