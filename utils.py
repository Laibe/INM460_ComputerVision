# Functions defined in FaceDetection.ipynb are repeated here for reuse in other notebooks via import utils

import numpy as np
from PIL import Image
import dlib
from pathlib import Path
import random
import torch

root = Path('./') # define root path
pretrained_cnn_detector = root.joinpath('mmod_human_face_detector.dat') # define pretrained cnn model path

mmod_hfd_path = root.joinpath('mmod_human_face_detector.dat')
if mmod_hfd_path.is_file() == False:
    raise ValueError('please download mmod_human_face_detector.dat from http://dlib.net/files/mmod_human_face_detector.dat.bz2 and unzip in current directory')
else:
    pass

hog_face_detect = dlib.get_frontal_face_detector()
cnn_face_detect = dlib.cnn_face_detection_model_v1(str(pretrained_cnn_detector.absolute()))

def load_image(file):
    """
    load file and return it as a numpy array 
    """
    img = Image.open(file)
    # We add this helper function to account for the iPhone SE quirk as explained in report
    try:
        if str(img.info['exif']).find('iPhone SE') != -1:
            img = img.rotate(-90,expand=1)
            return np.array(img)
        else:
            return np.array(img)
    except:
        print('No exif data found, continuing without check for iPhone SE')
        return np.array(img)

def face_detect(img,detector='hog',num_times_upsample=1):
    """
    img: numpy array 
    detector: 'hog' or 'cnn' (default: hog)
    num_times_upsample: upsampling makes the image bigger and allows for smaller faces to be detected (default: 1)
    returns bounding boxes
    """
    if detector == 'hog':
        return hog_face_detect(img,num_times_upsample)
    if detector == 'cnn':
        return cnn_face_detect(img,num_times_upsample)
    else:
        raise ValueError('Invalid detector selected. Valid options: hog,cnn')
        
        
def delib_rec_to_tuple(rec):
    """
    convert dlib rectangle object to standard python tuple
    """
    left = rec.left()
    right = rec.right()
    top = rec.top()
    bottom = rec.bottom()
    return (left,top,right,bottom)

        
 