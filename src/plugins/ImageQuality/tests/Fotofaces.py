import cv2
import sys

path = sys.argv[1]
image = cv2.imread(path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
score = cv2.quality.QualityBRISQUE_compute(gray, "brisque_model_live.yml", "brisque_range_live.yml")
obj = cv2.quality.QualityBRISQUE_create("brisque_model_live.yml", "brisque_range_live.yml")
score = obj.compute(gray)[0]
print(score)