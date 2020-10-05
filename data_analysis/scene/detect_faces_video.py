"""
Created on Tue Jun 16 16:41:07 2020

Usage: python detect_faces_video.py [file_name]

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
parser = argparse.ArgumentParser(description='Detects faces in a video')
parser.add_argument('-file_name', type=str, nargs=1,
                    help='video file name or webcam', default='webcam')

# Read the input arguments passed to the function and print them out
args = parser.parse_args()

if (args.file_name == 'webcam'):
    print('reading from: Webcam')
    cap = cv2.VideoCapture(0)
else:
    print('reading from: ', args.file_name[0])
    cap = cv2.VideoCapture(os.getcwd() + '/FaceDetectionData/' + args.file_name[0])
cap.set(3, 640) #WIDTH
cap.set(4, 480) #HEIGHT

fps = cap.get(cv2.CAP_PROP_FPS)
print('FPS: ', fps)
video_size = (640, 480)
fourcc = 'XVID'
#fourcc = 'FMP4'
out_video = cv2.VideoWriter(os.getcwd() + '/output_faces.avi',cv2.VideoWriter_fourcc(*fourcc), fps, video_size)

face_cascade = cv2.CascadeClassifier(os.getcwd() + '/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(os.getcwd() + '/haarcascade_eye.xml')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if (ret != True):
        break
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #print(len(faces))
    # Display the resulting frame
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('frame',frame)
    out_video.write(cv2.resize(frame, video_size, interpolation = cv2.INTER_AREA))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
out_video.release()
cap.release()
cv2.destroyAllWindows()

