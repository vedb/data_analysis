"""
Created on Tue Jun 16 10:22:18 2020

Usage: python detect_marker.py [start_time] [end_time]

        Parameters
        ----------

        start_time: tuple, len 2
            start time (min, sec) in the world video

        end_time: tuple, len 2
            end time (min, sec) in the world video

        data_path: str
            path to the world video

@author: KamranBinaee
"""

# import the necessary libraries
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
import sys
import argparse
import os
import pandas as pd


def read_input_arguments(args):

    # Eye video is recorded at 120 fps
    fps = 30
    # read the input arguments and store them to start and end time
    start_time = tuple(args.start_time)
    end_time = tuple(args.end_time)
    print('Start Time:', start_time)
    print('End Time:', end_time)

    start_index = (start_time[0] * 60 + start_time[1]) * fps
    end_index = (end_time[0] * 60 + end_time[1]) * fps
    print('Start Frame idx = %d' % start_index)
    print('End Frame idx= %d' % end_index)

    data_path = args.data_path[0]
    print(data_path)
    print('reading video: ', data_path + '/world.mp4')

    return start_index, end_index, data_path


def define_blob_detector():
    # Set our filtering parameters 
    # Initialize parameter settiing using cv2.SimpleBlobDetector 
    params = cv2.SimpleBlobDetector_Params() 
      
    # Set Area filtering parameters 
    params.filterByArea = True
    params.minArea = 500
    params.maxArea = 1800#700
      
    # Set Circularity filtering parameters 
    params.filterByCircularity = True 
    params.minCircularity = 0.4 # 0.7  
    # Set Convexity filtering parameters 
    params.filterByConvexity = True
    params.minConvexity = 0.7      
    # Set inertia filtering parameters 
    params.filterByInertia = True
    params.minInertiaRatio = 0.4 #0.6  

    # Create a detector with the parameters 
    return cv2.SimpleBlobDetector_create(params)

# Set the input arguments to the function and their types
parser = argparse.ArgumentParser(description='Detects the marker in the world video')
parser.add_argument('start_time', type=int, nargs=2,
                    help='start time (min, sec) in the eye video', default=(0, 20))
parser.add_argument('end_time', type=int, nargs=2,
                    help='end time (min, sec) in the eye video', default=(0, 50))
parser.add_argument('-data_path', type=str, nargs=1,
                    help='path to the eye video', default=os.getcwd()+'/EyeTrackingData')

# Read the input arguments passed to the function and print them out
args = parser.parse_args()
start_index, end_index, data_path = read_input_arguments(args)

# Prepare the empty lists for storing marker positions and the timestamps
marker_x = []
marker_y = []
marker_size = []
marker_index = []

# Instantiate the video capture from opencv to read the eye video, total number of frames, frame width and height
cap = cv2.VideoCapture(data_path + '/world.mp4')

numberOfFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print( 'Total Number of Frames: ', numberOfFrames )
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print('Frame Size :[', frame_height,frame_width, ']')


# Instantiate the video capture from imageio in order to read the eye video
vid = imageio.get_reader(data_path + '/world.mp4',  'ffmpeg')

scale = 0.5
size = (int(frame_width*scale), int(frame_height*scale))

# Instantiate the video recorder in order to store the processed images to an output video
fourcc = 'XVID'
out = cv2.VideoWriter('marker_detection_output.avi',cv2.VideoWriter_fourcc(*fourcc), 30, size)

# Here we set up our opencv blob detecter code
detector = define_blob_detector()

myString = '-'
# let's loop through the video from the start to the end index and detect where the marker is
for count in range(start_index, end_index):
    
    print("Progress: {0:.1f}% {s}".format(count*100/numberOfFrames, s = myString), end="\r", flush=True)

    # Read the next frame from the video.
    raw_image = vid.get_data(count)
    raw_image = cv2.resize(raw_image, None, fx = scale, fy = scale)
    # Switch the color channels since opencv reads the frames under BGR and the imageio uses RGB format
    raw_image[:, :, [0, 2]] = raw_image[:, :, [2, 0]]

    # Convert the image into grayscale
    gray = cv2.cvtColor(raw_image,cv2.COLOR_BGR2GRAY)
    
    # Perform image thresholding using a adaptive threshold window
    s = 63/(gray.shape[0]*scale)
    window_size = int(s*gray.shape[0]*scale) #11
    print(window_size, s,gray.shape[0], scale)
    binary_image = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,window_size,2)
    # Perform image erosion in order to remove the possible bright points inside the marker
    window_size = 3
    kernel = np.ones((window_size,window_size), int)
    binary_image = cv2.erode(binary_image, kernel, iterations = 1) # iterations = 2 for marker
    grey_3_channel = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    # Detect blobs using opencv blob detector that we setup earlier in the code
    keypoints = detector.detect(binary_image) 


    # Check if there is any blobs detected or not, if yes then draw it using a red color
    number_of_blobs = len(keypoints) 
    if number_of_blobs > 0:

        # Draw blobs on our image as red circles 
        blank = np.zeros((1, 1))
        blobs = cv2.drawKeypoints(raw_image, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Add text to image (Seems unncessary)
        #text = "# of Blobs: " + str(len(keypoints)) 
        #cv2.putText(blobs, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2) 
        for keypoint in keypoints:
            marker_x.append(keypoint.pt[0]*(1/scale))
            marker_y.append(keypoint.pt[1]*(1/scale))
            marker_size.append(keypoint.size)
            marker_index.append(count)
            #print("p = {:.1f} {:.1f} {:.1f} {}".format(keypoint.pt[0], keypoint.pt[1], keypoint.size, count))

        # Show blobs using opencv imshow method
        cv2.imshow('Frame',np.concatenate((blobs, grey_3_channel), axis=1))
        myString = '1'
        out.write(np.array(blobs))
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
        
    else:

        # If there is no blobs detected, just show the binary image and write it to the output video
        cv2.imshow('Frame',grey_3_channel)
        myString = '0'
        out.write(np.array(grey_3_channel))
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

print('\nDone!')
# Close all the opencv image frame windows opened
cv2.destroyAllWindows()

# Release the video writer handler so that the output video is saved to disk
out.release()

my_dict={'marker_x':np.array(marker_x), 'marker_y':np.array(marker_y),
        'marker_size':np.array(marker_size), 'marker_index':np.array(marker_index)}
data_frame = pd.DataFrame(my_dict)
data_frame.to_csv('detected_marker_positions.csv')
print('Saved marker data into csv file!')
print('\n\nThe end!')

'''
keypoint.pt[0],
keypoint.pt[1],
keypoint.size,
keypoint.angle,
keypoint.response,
keypoint.octave,
keypoint.class_id,
'''