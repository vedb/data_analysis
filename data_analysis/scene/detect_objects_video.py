import cv2
import numpy as np
import argparse
import os

# Set the input arguments to the function and their types
parser = argparse.ArgumentParser(description='Detects Objects in a video')
parser.add_argument('-file_name', type=str, nargs=1,
                    help='video file name or webcam', default='webcam')

# Read the input arguments passed to the function and print them out
args = parser.parse_args()

if (args.file_name == 'webcam'):
    print('reading from: webcam')
    cap = cv2.VideoCapture(0)
else:
    print('reading from: ', args.file_name[0])
    cap = cv2.VideoCapture(os.getcwd() + '/ObjectDetectionData/' + args.file_name[0])
cap.set(3, 640) #WIDTH
cap.set(4, 480) #HEIGHT

fps = cap.get(cv2.CAP_PROP_FPS)
print('FPS: ', fps)
video_size = (640, 480)
fourcc = 'XVID'
out_video = cv2.VideoWriter(os.getcwd() + 'detected_objects_output.avi',cv2.VideoWriter_fourcc(*fourcc), fps, video_size)

# Load Yolo
net = cv2.dnn.readNet("Yolo/yolov3.weights", "Yolo/yolov3.cfg")
classes = []
with open("Yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

minimum_probability = 0.5

while(cap.isOpened()):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Loading image
    #img = cv2.imread("room_ser.jpg")

    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    #print(height, width, channels)

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
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
                # Save the bouding box position and size to csv

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    #print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 1.5, color, 3)

    out_video.write(img)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out_video.release()
cap.release()
cv2.destroyAllWindows()
