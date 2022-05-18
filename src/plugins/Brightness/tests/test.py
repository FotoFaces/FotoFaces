import sys
import numpy as np
import cv2


path = sys.argv[1]
roi = cv2.imread(path)

hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
avg = np.mean(v)

print(avg)

# vicente_no_blur.jpg       167.889
# vicente.png               145.2664
# vieira.jpg                158.640
# neves.jpg                 167.005
# vicente_no_blur.jpg       138.103
# vicente_no_blur.jpg 167.889

 