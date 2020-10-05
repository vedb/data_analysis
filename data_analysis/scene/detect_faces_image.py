"""
Created on Tue Jun 16 16:41:07 2020

Usage: python detect_faces_image.py [file_name]

        Parameters
        ----------

        file_name: str
            image file name

@author: KamranBinaee
"""

import cv2
import numpy as np
import argparse
import os

# Set the input arguments to the function and their types
parser = argparse.ArgumentParser(description='Detects faces in an image')
parser.add_argument('file_name', type=str, nargs=1,
                    help='input image file name', default='Ellen-Selfie.jpg')

# Read the input arguments passed to the function and print them out
args = parser.parse_args()

face_cascade = cv2.CascadeClassifier(os.getcwd() + '/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(os.getcwd() + '/haarcascade_eye.xml')

file_name = args.file_name[0]
print('Reading: ', os.getcwd() + '/FaceDetectionData/' + file_name)
img = cv2.imread(os.getcwd() + '/FaceDetectionData/' + file_name)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, 1.05, 3, 10) # 1.3, 5
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    # Detect the eyes in the face bounding box
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.05, 3, 10)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img',img)
while cv2.waitKey(0) & 0xFF == ord('q'):
    break

# Save the output image with detected faces
cv2.imwrite('output_'+ file_name, img)
cv2.destroyAllWindows()
print('Done!')