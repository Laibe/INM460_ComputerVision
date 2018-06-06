# import utils function
from utils import load_image,face_detect,delib_rec_to_tuple,load_image

# import external libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from scipy.cluster.vq import *
import PIL
from PIL import Image,ImageDraw
import scipy.io
from sklearn.externals import joblib



model_dir = 'models' 
    
# data transformations should be identical to the one used during training.

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}


# define class_names
class_names = ['001',
 '002',
 '003',
 '004',
 '005',
 '006',
 '007',
 '008',
 '009',
 '010',
 '011',
 '012',
 '013',
 '014',
 '015',
 '016',
 '017',
 '018',
 '035',
 '037',
 '045',
 '046',
 '050',
 '051',
 '052',
 '053',
 '054',
 '055',
 '056',
 '057',
 '058',
 '059',
 '060',
 '061',
 '062',
 '063',
 '064',
 '065',
 '066',
 '067',
 '068',
 '069',
 '070',
 '107',
 '108',
 '161',
 '164',
 '165',
 '166',
 '167',
 '168',
 '169',
 '170']
class_names_int = np.array(list(map(int, class_names))) # convert to integer


def face_extractor_to_torchtensor(image,num_times_upsample=1,detector='hog'):
    """
    Args: 
        image: RGB image (e.g. /path/image.jpg )
    Returns: a tensor containing collected faces, their bounding boxes and their central face regions
    (collected_face_tensors,bboxes,central_face_regions)
    """
    numpyimage = load_image(image) # load image as an array
    face_bboxes = face_detect(numpyimage,detector,num_times_upsample) # detect faces
    collected_face_tensors = torch.Tensor(1,3,224,224) #  empty 4d tensor
    bboxes = [] 
    central_face_regs = [] 
    for counter,f in enumerate(face_bboxes):
            left,top,right,bottom = delib_rec_to_tuple(f) # get coordinates 
            bboxes.extend([left,top,right,bottom]) # extend list with found coordinates
            x = int((left+right)/2) # calculate central face regions
            y = int((top+bottom)/2)
            central_face_regs.extend([x,y]) # extend list with found coordinates
            face_pil = Image.fromarray(numpyimage[top:bottom,left:right]) # create pillow image
            face_tensor = data_transforms['val'](face_pil); # do data transformation
            face_tensor.unsqueeze_(0); # unsqueeze
            collected_face_tensors = torch.cat([collected_face_tensors,face_tensor]) # cat to tensor
    return collected_face_tensors[1:],bboxes,central_face_regs


def face_extractor_to_list(image,num_times_upsample=1,detector='hog'):
    """
    Args:
        image: RGB image (e.g. /path/image.jpg )
    Returns: outputs a tensor containing collected faces, their bounding boxes and their central face regions
    (collected_face_tensors,bboxes,central_face_regions)
    """
    numpyimage = load_image(image)
    face_bboxes = face_detect(numpyimage,detector,num_times_upsample)
    collected_face_list = [] 
    bboxes = []
    central_face_regs = []
    for counter,f in enumerate(face_bboxes):
            left,top,right,bottom = delib_rec_to_tuple(f) # get coordinates as shown above
            bboxes.extend([left,top,right,bottom])  # extend list with found coordinates
            x = int((left+right)/2) # calculate central face regions 
            y = int((top+bottom)/2)
            central_face_regs.extend([x,y]) # extend list with found coordinates
            face = numpyimage[top:bottom,left:right] # get face
            collected_face_list.append(face) # append found faces to list
    return collected_face_list,bboxes,central_face_regs



def detect_and_compute_inference(facelist,featuretype):
    '''
    detects keypoints and computes descriptors
    Args:
        list of faces, e.g. output from the function face_extractor_to_list 
        featuretype: SURF,SIFT
    Returns: 
         array with descriptions, 64 columns (elements) for SURF and 128 for SIFT
    '''
    # Detect, compute and return all features found on images
    descriptions = []
    if featuretype == 'SURF':
        detector = cv2.xfeatures2d.SURF_create()
    elif featuretype == 'SIFT':
        detector = cv2.xfeatures2d.SIFT_create()
    else:
        raise ValueError('invalid featuretype selected, valid: SURF, SIFT')
    for face in facelist:
        grayface = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        grayface = cv2.resize(grayface,(224,224))
        keypoints, description = detector.detectAndCompute(grayface, None)
        descriptions.append(description)
    return descriptions


def bag_of_features_inference(descriptions,vocabulary,vocabulary_size):
    '''
    Args:
        descriptions: obtained from the detect_and_compute_inference function
        vocabulary
        vocubuarly_size
    Returns:
        bag of image features
    '''
    set_size = len(descriptions)
    im_features = np.zeros((set_size, vocabulary_size), "float32")
    for i, descriptor in enumerate(descriptions):
        words, _ = vq(descriptor, vocabulary)
        for w in words:
            im_features[i][w] +=1          
    return im_features


def load_cnn_model(modelname,architecture):
    """
    Args:
        modelname: e.g. ResNet34YYYYMMDD.pth
        architecture: resnet34
    """
    if architecture == 'resnet34':
        model_ft = models.resnet34(pretrained=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 53)
        model_checkpoint = torch.load(os.path.join(model_dir, modelname), map_location=lambda storage, loc: storage)
        model_ft.load_state_dict(model_checkpoint)
    return model_ft

def load_bof_model(modelname):
    '''
    loads a pickle containing model, class_names, standard scaler, vocabulary size and vocabulary
    '''
    model, class_names, stdSlr, vocabulary_size, voc = joblib.load(os.path.join(model_dir, modelname))
    return model, class_names, stdSlr, vocabulary_size, voc

       
def create_report_matrix(central_face_regions,predarray):
    '''
    Args:
        central_face_regions: list
        predarry: numpy array
    Returns: np array Nx3 where N is equal to the number of faces detect in the image
    '''
    
    cf_array = np.array(central_face_regions).reshape(int(len(central_face_regions)/2),2) # converts list to array 2xN
    pred_classes = class_names_int[predarray] # get the class names 
    return np.column_stack( (pred_classes,cf_array) )


def create_report_matrix_bof(central_face_regions,predarray):
    '''
    Args:
        central_face_regions: list
        predarry: numpy array
    Returns: np array Nx3 where N is equal to the number of faces detect in the image
    '''
    
    cf_array = np.array(central_face_regions).reshape(int(len(central_face_regions)/2),2) # converts list to array 2xN
    return np.column_stack( (predarray,cf_array) )


def RecogniseFace(I,featureType,classifierName,faceDetector='hog'):
    '''
    Args:
        I: image to classify
        feature type: SIFT,SURF
        classifierName: SVM, LogisticRegression,ResNet34
        faceDetector: hog,cnn
    Returns: 
        report: np array Nx3 where N is equal to the number of faces detect in the image
    '''
    if classifierName == 'ResNet34':
        # extract the bboxes, central face regions and store bboxes in a pytorch tensor (faces_from_image_tensors)
        faces_from_image_tensors,bboxes,central_face_regs = face_extractor_to_torchtensor(I,detector=faceDetector)
        
        model_ft = load_cnn_model('ResNet34_20180315.pth','resnet34') # load the model 
        model_ft.eval(); # set to evaluation mode
        inputs = Variable(faces_from_image_tensors) # convert tensors to Variables
        outputs = model_ft(inputs) # do the forward pass
        out, preds = torch.max(outputs.data, 1) # return the predictions
        predarray = preds.numpy()
        report_matrix = create_report_matrix(central_face_regs,predarray) # create the report matrix in the required format
    
    elif classifierName == 'SVM' and featureType == 'SURF':
        faces_from_image_list, bboxes, central_face_regs = face_extractor_to_list(I,detector=faceDetector)
        model, class_names, stdSlr, vocabulary_size, voc = load_bof_model('SVC_SURF_gs_20180309.pkl')
        desc = detect_and_compute_inference(faces_from_image_list,featureType)
        im_bof = bag_of_features_inference(desc,voc,vocabulary_size)
        im_bof = stdSlr.transform(im_bof)
        pred = model.predict(im_bof)
        predarray = np.array(list(map(int, pred)))
        report_matrix = create_report_matrix_bof(central_face_regs,predarray)
        
    elif classifierName == 'SVM' and featureType == 'SIFT':
        faces_from_image_list, bboxes, central_face_regs = face_extractor_to_list(I,detector=faceDetector)
        model, class_names, stdSlr, vocabulary_size, voc = load_bof_model('SVC_SIFT_gs_20180310.pkl')
        desc = detect_and_compute_inference(faces_from_image_list,featureType)
        im_bof = bag_of_features_inference(desc,voc,vocabulary_size)
        im_bof = stdSlr.transform(im_bof)
        pred = model.predict(im_bof)
        predarray = np.array(list(map(int, pred)))
        report_matrix = create_report_matrix_bof(central_face_regs,predarray)
    
    elif classifierName == 'LR' and featureType == 'SURF':
        faces_from_image_list, bboxes, central_face_regs = face_extractor_to_list(I,detector=faceDetector)
        model, class_names, stdSlr, vocabulary_size, voc = load_bof_model('LOGREG_SURF_gs_20180310.pkl')
        desc = detect_and_compute_inference(faces_from_image_list,featureType)
        im_bof = bag_of_features_inference(desc,voc,vocabulary_size)
        im_bof = stdSlr.transform(im_bof)
        pred = model.predict(im_bof)
        predarray = np.array(list(map(int, pred)))
        report_matrix = create_report_matrix_bof(central_face_regs,predarray)
        
        
    elif classifierName == 'LR' and featureType == 'SIFT':
        faces_from_image_list, bboxes, central_face_regs = face_extractor_to_list(I,detector=faceDetector)
        model, class_names, stdSlr, vocabulary_size, voc = load_bof_model('LOGREG_SIFT_gs_20180310.pkl')
        desc = detect_and_compute_inference(faces_from_image_list,featureType)
        im_bof = bag_of_features_inference(desc,voc,vocabulary_size)
        im_bof = stdSlr.transform(im_bof)
        pred = model.predict(im_bof)
        predarray = np.array(list(map(int, pred)))
        report_matrix = create_report_matrix_bof(central_face_regs,predarray)
     
    else:
        raise ValueError('Not a valid feature/classifier combination')
                             
    return report_matrix
    

def RecogniseFace_ReturnAnnotatedImage(I,featureType,classifierName,faceDetector='hog'):
    '''
    Args:
        I: image to classify
        feature type: SIFT,SURF
        classifierName: SVM, LogisticRegression,ResNet34
        faceDetector: hog,cnn
    Returns: 
        report: np array Nx3 where N is equal to the number of faces detect in the image
        image: annotated image with bounding boxes around the faces and their predicted class id 
    '''
    if classifierName == 'ResNet34':
        model_ft = load_cnn_model('ResNet34_20180315.pth','resnet34')
        faces_from_image_tensors,bboxes,central_face_regs = face_extractor_to_torchtensor(I,detector=faceDetector)
        model_ft.eval();
        inputs = Variable(faces_from_image_tensors)
        outputs = model_ft(inputs)
        out, preds = torch.max(outputs.data, 1)
        predarray = preds.numpy()
        report_matrix = create_report_matrix(central_face_regs,preds)
     
    
    elif classifierName == 'SVM' and featureType == 'SURF':
        faces_from_image_list, bboxes, central_face_regs = face_extractor_to_list(I,detector=faceDetector)
        model, class_names, stdSlr, vocabulary_size, voc = load_bof_model('SVC_SURF_gs_20180309.pkl')
        desc = detect_and_compute_inference(faces_from_image_list,featureType)
        im_bof = bag_of_features_inference(desc,voc,vocabulary_size)
        im_bof = stdSlr.transform(im_bof)
        pred = model.predict(im_bof)
        predarray = np.array(list(map(int, pred)))
        report_matrix = create_report_matrix_bof(central_face_regs,predarray)
        
        
    elif classifierName == 'SVM' and featureType == 'SIFT':
        faces_from_image_list, bboxes, central_face_regs = face_extractor_to_list(I,detector=faceDetector)
        model, class_names, stdSlr, vocabulary_size, voc = load_bof_model('SVC_SIFT_gs_20180310.pkl')
        desc = detect_and_compute_inference(faces_from_image_list,featureType)
        im_bof = bag_of_features_inference(desc,voc,vocabulary_size)
        im_bof = stdSlr.transform(im_bof)
        pred = model.predict(im_bof)
        predarray = np.array(list(map(int, pred)))
        report_matrix = create_report_matrix_bof(central_face_regs,predarray)
    
    elif classifierName == 'LR' and featureType == 'SURF':
        faces_from_image_list, bboxes, central_face_regs = face_extractor_to_list(I,detector=faceDetector)
        model, class_names, stdSlr, vocabulary_size, voc = load_bof_model('LOGREG_SURF_gs_20180310.pkl')
        desc = detect_and_compute_inference(faces_from_image_list,featureType)
        im_bof = bag_of_features_inference(desc,voc,vocabulary_size)
        im_bof = stdSlr.transform(im_bof)
        pred = model.predict(im_bof)
        predarray = np.array(list(map(int, pred)))
        report_matrix = create_report_matrix_bof(central_face_regs,predarray)
        
        
    elif classifierName == 'LR' and featureType == 'SIFT':
        faces_from_image_list, bboxes, central_face_regs = face_extractor_to_list(I,detector=faceDetector)
        model, class_names, stdSlr, vocabulary_size, voc = load_bof_model('LOGREG_SIFT_gs_20180310.pkl')
        desc = detect_and_compute_inference(faces_from_image_list,featureType)
        im_bof = bag_of_features_inference(desc,voc,vocabulary_size)
        im_bof = stdSlr.transform(im_bof)
        pred = model.predict(im_bof)
        predarray = np.array(list(map(int, pred)))
        report_matrix = create_report_matrix_bof(central_face_regs,predarray)
    
    else:
        raise ValueError('Not a valid feature/classifier combination')
    
    
    # Draw the boxes and annotations 
    imgpil = Image.fromarray(load_image(I))
    draw = ImageDraw.Draw(imgpil)
    bboxes_array= np.array(bboxes).reshape(int(len(bboxes)/4),4)
    for i,(left,top,right,bottom) in enumerate(bboxes_array):
### Start of external code block ###
# https://github.com/ageitgey/face_recognition/blob/master/examples/identify_and_draw_boxes_on_faces.py
        draw.rectangle(((left, top),(right, bottom)), outline=(0, 255, 0))
        text_width, text_height = draw.textsize(str(report_matrix[i][0]))
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 255, 0), outline=(0, 255, 0))
        draw.text((left + 6, bottom - text_height - 5), str(report_matrix[i][0]), fill=(0, 0, 0, 0))
### End of external code block ### 
    del draw
    return report_matrix,imgpil