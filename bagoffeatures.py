# Parts of this code are inspired from @hugos94 implementation: https://github.com/hugos94/bag-of-features/blob/master/learn.py
from scipy.cluster.vq import *
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import numpy as np
import cv2

def detect_and_compute(train_path,featuretype):
    '''
    detects keypoints and computes descriptors
    input:
        train_path: path to training images, where dictionar name = class
        featuretype: SURF,SIFT
    ouput: 
        descriptions: ('path/to/image',array[])
         array has 64 columns (elemnts) for SURF and 128 for SIFT
    '''
    pattern = '*'
    # Detect, compute and return all features found on images
    descriptions = []
    image_classes = []
    if featuretype == 'SURF':
        detector = cv2.xfeatures2d.SURF_create()
    elif featuretype == 'SIFT':
        detector = cv2.xfeatures2d.SIFT_create()
    else:
        raise ValueError('invalid featuretype selected, valid: SURF, SIFT')
    for child in train_path.iterdir():
        for image_path in child.glob(pattern):
            class_id = str(child.absolute()).split('/')[-1]
            image = cv2.imread(str(image_path.absolute()),0)
            image = cv2.resize(image,(224,224))
            keypoints, description = detector.detectAndCompute(image, None)
            descriptions.append((str(image_path.absolute()),description))
            image_classes.append(class_id)
    return descriptions,image_classes

def stack_descriptors(descr,featuretype):
    '''
    input:
        descr: descriptions in tuples obtained from detect_and_compute and stacks them vertically
    output:
        stacked descriptor
    '''
    if featuretype == 'SURF':
        elementsinarray = 64
    elif featuretype == 'SIFT':
        elementsinarray = 128
    else:
        raise ValueError('invalid featuretype selected, valid: SURF, SIFT')
    
    stacked_descriptors = np.array([], dtype=np.float32).reshape(0,elementsinarray)
    for _, d in descr:
        stacked_descriptors = np.vstack((stacked_descriptors,d))
    return stacked_descriptors

def create_vocabulary(stacked_descriptors,vocabulary_size=500):
    '''
    creates a vocabluary with kmeans
    input: 
        stacked_descriptors: output from Stack_descriptors function
        vocabulary_size: default 500 
    output: 
        vocabulary
    '''
    voc, _ = kmeans(stacked_descriptors, vocabulary_size, 1)
    return voc
    
def bag_of_features(descriptions,vocabulary,vocabulary_size):
    '''
    input:
        descriptions: obtained from the detect_and_compute function
        vocabulary: output from create_vocabulary
        vocubuarly_size
    output:
        bag of image features
        standard scaler
    '''
    set_size = len(descriptions)
    im_features = np.zeros((set_size, vocabulary_size), "float32")
    for i,(_ , descriptor) in enumerate(descriptions):
        words, _ = vq(descriptor, vocabulary)
        for w in words:
            im_features[i][w] +=1          
    return im_features
    

