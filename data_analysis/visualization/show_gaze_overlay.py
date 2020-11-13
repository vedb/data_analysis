"""
Created on Tue Jun 16 10:22:18 2020

Usage: python show_gaze_overlay.py [start_time] [end_time]

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

    # World video is recorded at 30 fps
    fps = 30
    # read the input arguments and store them to start and end time
    start_time = tuple(args.start_time)
    end_time = tuple(args.end_time)
    print("Start Time:", start_time)
    print("End Time:", end_time)

    start_index = (start_time[0] * 60 + start_time[1]) * fps
    end_index = (end_time[0] * 60 + end_time[1]) * fps
    print("Start Frame idx = %d" % start_index)
    print("End Frame idx= %d" % end_index)

    data_path = args.data_path
    print("reading video: ", data_path + "/eye0.mp4")

    return start_index, end_index, data_path


# Set the input arguments to the function and their types
parser = argparse.ArgumentParser(description="Detects the pupil in the eye video")
parser.add_argument(
    "start_time",
    type=int,
    nargs=2,
    help="start time (min, sec) in the eye video",
    default=(0, 20),
)

parser.add_argument(
    "end_time",
    type=int,
    nargs=2,
    help="end time (min, sec) in the eye video",
    default=(0, 50),
)

parser.add_argument(
    "-data_path",
    type=str,
    nargs=1,
    help="path to the eye video",
    default=os.getcwd() + "/EyeTrackingData",
)

# Read the input arguments passed to the function and print them out
args = parser.parse_args()
start_index, end_index, data_path = read_input_arguments(args)

horizontal_pixels = 1280
vertical_pixels = 1024

scale_x = 0.5
scale_y = 0.5
sub_folder = "/000/"

# Todo: Pass fps as an argument
fps = 30

# gaze_data_frame = pd.read_csv(fpath + 'exports' + sub_folder + 'gaze_positions.csv')
gaze_data_frame = pd.read_csv(data_path + "/gaze_positions.csv")
world_time_stamps = np.load(data_path + "/world_timestamps.npy")

camera = "world"
vid = imageio.get_reader(data_path + "/world.mp4", "ffmpeg")
cap = cv2.VideoCapture(data_path + "/world.mp4")

frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

size = (frame_width, frame_height)
print("frame size:", size)
# start_index = (start[0] * 60 + start[1]) * fps
# end_index = (end[0] * 60 + end[1]) * fps
print("First Frame = %d" % start_index)
print("Last Frame = %d" % end_index)

print("scale[x,y] = ", scale_x, scale_y)

# Instantiate the video recorder in order to store the processed images to an output video
fourcc = "XVID"
out = cv2.VideoWriter(
    "gaze_overlay_output.avi", cv2.VideoWriter_fourcc(*fourcc), 30, size
)

# Read the next frame from the video.
for i in range(start_index, end_index):

    # img = imstack[i,:,:,:]
    img = vid.get_data(i)
    img[:, :, [0, 2]] = img[:, :, [2, 0]]
    img = cv2.resize(img, None, fx=scale_x, fy=scale_y)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gaze_index = np.argmin(
        np.abs(
            (gaze_data_frame.gaze_timestamp.values - world_time_stamps[i]).astype(float)
        )
    )
    gaze_norm_x = gaze_data_frame.norm_pos_x.values[gaze_index]
    gaze_norm_y = gaze_data_frame.norm_pos_y.values[gaze_index]

    gaze_pixel_x = int(gaze_norm_x * horizontal_pixels * scale_x)
    gaze_pixel_y = int((1 - gaze_norm_y) * vertical_pixels * scale_y)
    frame_no_gaze = img.copy()
    img = cv2.circle(img, (gaze_pixel_x, gaze_pixel_y), 10, (255, 255, 0), 8)

    # Todo: Create a separate function for adding text info on the frame
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (10, 30)
    # fontScale
    font_scale = 0.5
    # Blue color in BGR
    color = (0, 255, 255)
    # Line thickness of 2 px
    thickness = 1
    text = "confidence: " + str(
        np.round(gaze_data_frame.confidence.values[gaze_index], 2)
    )
    # Using cv2.putText() method
    img = cv2.putText(img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

    w = 50
    h = 50
    x_min = gaze_pixel_x - w
    x_max = gaze_pixel_x + w

    y_min = gaze_pixel_y - h
    y_max = gaze_pixel_y + h

    if gaze_pixel_x - w < 0:
        x_min = 0
        x_max = 2 * w
    if gaze_pixel_x + w >= (horizontal_pixels * scale_x):
        x_min = horizontal_pixels * scale_x - 2 * w - 1
        x_max = horizontal_pixels * scale_x - 1

    if gaze_pixel_y - h < 0:
        y_min = 0
        y_max = 2 * h
    if gaze_pixel_y + h >= (vertical_pixels * scale_y):
        y_min = vertical_pixels * scale_y - 2 * h - 1
        y_max = vertical_pixels * scale_y - 1

    range_x = np.arange(int(x_min), int(x_max))
    range_y = np.arange(int(y_min), int(y_max))

    fovea = frame_no_gaze[int(y_min) : int(y_max), int(x_min) : int(x_max), :]
    print(frame_no_gaze.shape, fovea.shape)
    # print(range_x.shape, range_y.shape)
    # print(range_x.min(), range_x.max(), range_y.min(), range_y.max())
    cv2.imshow("img", img)
    out.write(img)
    cv2.imshow("fovea", fovea)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

print("\nDone!")

cv2.destroyAllWindows()
# Release the video writer handler so that the output video is saved to disk
out.release()
