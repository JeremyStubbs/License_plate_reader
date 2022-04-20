from turtle import color
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras_utils import detect_lp
from os.path import splitext,basename
from keras.models import model_from_json
import glob
from utils import im2single
from label import Shape, writeShapes


def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)

wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)

lp_threshold=0.5

def get_plate(image_path, Dmax=608, Dmin=256):
    Ivehicle = image_path
    ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
    side  = int(ratio*288.)
    bound_dim = min(side + (side%(2**4)),608)

    Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)

    return Llp, LlpImgs

cap = cv2.VideoCapture(0)

while True:
    success,img = cap.read()
    
    coord, color = get_plate(img)

    if len(coord):
        h, w, c = img.shape
        x = coord[0].pts
        start_point = (int(w*min(x[0])), int(h*min(x[1])))
        end_point = (int(w*max(x[0])),  int(h*max(x[1])))

        cv2.rectangle(img, start_point, end_point, color = (0,255,0), thickness=2)
        plt.imshow(img)
 
    cv2.imshow("output",img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()