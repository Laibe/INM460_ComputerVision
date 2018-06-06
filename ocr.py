from utils import load_image
# load external libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
from imutils.perspective import four_point_transform
import pytesseract
from PIL import Image,ImageDraw
from collections import Counter

def detectFileFormat(filename):
    imgend = ['jpg','JPG','png','PNG','JPEG','jpeg']
    movend = ['mov','MOV','mp4']
    if filename.split('.')[-1] in imgend:
        fileformat = 'img'
    elif filename.split('.')[-1] in movend:
        fileformat = 'mov'
    else:
        raise ValueError('invalid file-format')
    return fileformat

def ocr(arrayimage):
    img = Image.fromarray(arrayimage) # PIL image
    return pytesseract.image_to_string(img)

def detectNumImage(img,returnimage=False):
    '''
    img: in array format
    '''
    listofhull = []
    numbers = []
    
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)  # resize for faster computation
    
### Start of external code block ### 
# https://stackoverflow.com/questions/10533233/opencv-c-obj-c-advanced-square-detection
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)   # convert to gray scale
    ret,thresh = cv2.threshold(gray,190,255,0)    # threshold to highlight white A4 paper
    contours = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) # find contours
    contours = contours[1] 
 
    for cnt in contours: # loop through found contours
        if cv2.contourArea(cnt)>1000:     # ignore smaller areas
            hull = cv2.convexHull(cnt)    # find the convex hull of contour
            hull = cv2.approxPolyDP(hull,0.02*cv2.arcLength(hull,True),True) # approximate polygonal curves

            if len(hull)==4:    # find rectangles with four corners       
### End of external code block ### 
                    listofhull.append(hull)

    for h in listofhull: # for each found rectangle run ocr
        warped = four_point_transform(gray, h.reshape(4, 2)) # transform perspective
        h,w = warped.shape # get dimension
        hc = (int(h - h/6),int(h/6)) # crop
        wc = (int(w - w/6),int(w/6)) 
        warped = warped[hc[1]:hc[0], wc[1]:wc[0]]
        ocr_raw = ocr(warped)
        ocr_corrected = ocr_raw.replace('O','0') 
        numbers.append(ocr_corrected)
    
    numbers_clean = [n for n in numbers if n.isdigit()]
    if numbers_clean == []: # if no numbers were found try the sliding window approach (takes long)
        print('No numbers were found in the detected rectangles, now trying the sliding window approach. This may take a minute or two.')
        numbers_clean = detectNumImageSlidingWindow(img)
        if returnimage == True: # since no rectangles were found we can't return an annotated image
            return numbers_clean,img # return original image
        else:
            return numbers_clean
    
    else:    
        hullnumber = list(map(bool, numbers)).index(True) # returns the index of the hull where we found a number    
        numbers = list(filter(None, numbers))

        if returnimage == True:
            if numbers_clean != []:
                cv2.drawContours(img,[listofhull[hullnumber]],0,(0,255,0),2)
                rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return numbers_clean,img

        else:
            return numbers_clean
        
        
        
def detectNumVideo(filename):
    '''
    function that detects the white A4 paper with the written number on it and recognises which number it is. 
    '''

    cap = cv2.VideoCapture(filename)   # read video
    collectednumbers = []
    counter = Counter  # counter to determine most frequent numbers
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret == True:

            # For every frame do the following operations:
            
            listofhull = [] 
            numbers = []
### Start of external code block ### 
# https://stackoverflow.com/questions/10533233/opencv-c-obj-c-advanced-square-detection 
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret,thresh = cv2.threshold(gray,190,255,0)

            contours = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) # find contours
            contours = contours[1] 

            for cnt in contours: # loop through found contours
                if cv2.contourArea(cnt)>1000:     # ignore smaller areas
                    hull = cv2.convexHull(cnt)    # find the convex hull of contour
                    hull = cv2.approxPolyDP(hull,0.02*cv2.arcLength(hull,True),True) # approximate polygonal curves

                    if len(hull)==4:    # find rectangles with four corners
### End of external code block ### 
                            listofhull.append(hull)

            for h in listofhull: # for each found rectangle run ocr
                warped = four_point_transform(gray, h.reshape(4, 2)) # transform perspective
                h,w = warped.shape # get dimension
                hc = (int(h - h/6),int(h/6)) # crop to get rid of hands and fingers
                wc = (int(w - w/6),int(w/6)) 
                warped = warped[hc[1]:hc[0], wc[1]:wc[0]]
                ocr_raw = ocr(warped)
                ocr_corrected = ocr_raw.replace('O','0') 
                numbers.append(ocr_corrected)
            numbers_clean = [n for n in numbers if n.isdigit()]

           # returns the index of the hull where we found a number    
            collectednumbers.extend(numbers_clean) # append list of numbers
        else:
            cap.release()
            break

        # When everything done, release the capture and count the numbers
    cap.release()
    if collectednumbers == []:
        return []
    else:
        cn = counter(collectednumbers)
        mostcommon = cn.most_common()[0][0]   # return the number with most occurances as this is the most probably number to be in the video
        return mostcommon

    
### Start of external code block ###
# https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/ 
def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
            
### End of external code block ###

def detectNumImageSlidingWindow(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (0,0), fx=0.5, fy=0.5)
    winW = int(0.10* resized.shape[0])
    winH = winW
    numbers = []
### Start of external code block ###
# https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/ 
    for (x, y, window) in sliding_window(resized, stepSize=64, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
### End of external code block ###          
        wimg = resized[y:y+winH,x:x+winW]# the image which has to be predicted
        num = ocr(wimg)
        numbers.append(num)
    numbers_clean = [n for n in numbers if n.isdigit()] # only consider digits
    numbers_clean2 = [n for n in numbers_clean if len(n)==2] # only consider digits with length two
    counter = Counter
    if numbers_clean2 == []:
        return []
    else:
        cn = counter(numbers_clean2)
        mostcommon = cn.most_common()[0][0]  
        
    return [mostcommon]
        
          


def detectNum(filename,returnimage=False):
    '''
    Function that detects the white A4 paper and recognises the number written on it
    input: image or movie
    supported formats: ['jpg','JPG','png','PNG','JPEG','jpeg','mov','MOV','mp4']
    returnimage: optional switch that returns the image with a bounding box around the paper
    this function is not available for video formats
    '''
    fileformat = detectFileFormat(filename)
    if fileformat == 'img' and returnimage == False:
        image = load_image(filename)
        num = detectNumImage(image,False)
        return num
    elif fileformat == 'img' and returnimage == True:
        image = load_image(filename)
        num,img = detectNumImage(image,True)
        return num,img
    
    elif fileformat == 'mov':
        num = detectNumVideo(filename)
        return num
    else:
        raise ValueError('check file-format')