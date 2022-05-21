import base64
import json
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg  
#from PIL import Image
import io     
import sys         
import cv2

import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

api = 'http://127.0.0.1:5000/'
reference = sys.argv[1]
candidate = sys.argv[2]

with open(reference, "rb") as f:
    ref_bytes = f.read()
ref_b64 = base64.b64encode(ref_bytes).decode("utf8")

with open(candidate, "rb") as f:
    can_bytes = f.read()
can_b64 = base64.b64encode(can_bytes).decode("utf8")

mp_encoder = MultipartEncoder(
    fields={
        'candidate': can_b64 ,
        'id':'1234',
        'reference':ref_b64,
    }
)
r = requests.post(
    api,
    data=mp_encoder,  # The MultipartEncoder is posted as data, don't use files=...!
    # The MultipartEncoder provides the content-type header with the boundary:
    headers={'Content-Type': mp_encoder.content_type}
)
try:
    data = r.json()
    image_data = data["cropped"]
    image_data = base64.b64decode(image_data)
    with open(reference, 'wb') as f:
        f.write(image_data)
    image = cv2.imread(reference)
    cv2.imshow("image", image)()
    cv2.waitKey(0)
    #decode_cropped = base64.b64decode(data['cropped']).deco
    #img = mpimg.imread(data['cropped'])
    #imgplot = plt.imshow(img)
    #plt.show()  
    print(data['feedback'])                
except requests.exceptions.RequestException:
    print(r.text)