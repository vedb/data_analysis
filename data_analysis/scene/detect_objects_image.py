"""
Created on Tue Jun 16 16:41:07 2020

Usage: python detect_objects_image.py [file_name]

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
parser = argparse.ArgumentParser(description='Detects objects in an image')
parser.add_argument('file_name', type=str, nargs=1,
                    help='input image file name')

# Read the input arguments passed to the function and print them out
args = parser.parse_args()

# Loading image
file_name = args.file_name[0]
print('Reading: ', os.getcwd() + '/ObjectDetectionData/' + file_name)
img = cv2.imread(os.getcwd() + '/ObjectDetectionData/' + file_name)
#img = cv2.resize(img, None, fx=0.8, fy=0.8)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width, channels = img.shape

# Load Yolo
net = cv2.dnn.readNet("Yolo/yolov3.weights", "Yolo/yolov3.cfg")
classes = []
with open("Yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

# fontScale 
fontScale = 1
   
# Line thickness of 2 px 
thickness = 1

total_object_count = 5
minimum_probability = 0.5
# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[total_object_count:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > minimum_probability:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#print(indexes)
print('Detected Objects: ')
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        print(label, np.round(confidences[i],2))
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 1.5, color, 2, cv2.LINE_AA)

cv2.imshow("Image", img)
while (cv2.waitKey(0) & 0xFF == ord('q')):
    cv2.destroyAllWindows()
# Save the output image with detected objects
cv2.imwrite('output_'+ file_name, img)
print('Done!')