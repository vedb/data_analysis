import numpy as np
import cv2
import pandas as pd


def detect_checkerboard(path, checkerboard_size, scale, start_seconds, end_seconds):

    horizontal_pixels = 1280
    vertical_pixels = 1024

    rows = checkerboard_size[0]
    cols = checkerboard_size[1]
    scale_x = scale[0]
    scale_y = scale[1]
    sub_folder = '/000/'

    # Todo: Pass fps as an argument
    fps = 30
    safe_margin = 5

    image_list = []
    marker_found_index = []

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    gaze_data_frame = pd.read_csv(path + 'exports' + sub_folder + 'gaze_positions.csv')
    world_time_stamps = np.load(path + 'world_timestamps.npy')

    print('reading video: ', path + 'world.mp4')
    cap = cv2.VideoCapture(path + 'world.mp4')
    frame_numbers = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Total Number of Frames: ', frame_numbers)

    start_index = (start_seconds + safe_margin) * fps
    end_index = (end_seconds - safe_margin) * fps
    print('First Frame = %d' % start_index)
    print('Last Frame = %d' % end_index)

    print('scale[x,y] = ', scale_x, scale_y)

    my_string = '-'
    for count in range(0, frame_numbers):

        print("Progress: {0:.1f}% {s}".format(count * 100 / frame_numbers, s=my_string), end="\r", flush=False)
        if count < start_index:
            ret, frame = cap.read()
            continue
        elif count > end_index:
            break
        else:

            # Read the next frame from the video. If you set frame 749 above then the code will return the last frame.
            ret, img = cap.read()
            img = cv2.resize(img, None, fx=scale_x, fy=scale_y)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (6, 8), None)

            # If found, add object points, image points (after refining them)
            if ret is True:
                # print('====> Found [%d]!' %(count))
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                # Draw and display the corners
                img_1 = cv2.drawChessboardCorners(img, (6, 8), corners2, ret)

                gaze_index = np.argmin(
                    np.abs((gaze_data_frame.gaze_timestamp.values - world_time_stamps[count]).astype(float)))
                gaze_norm_x = gaze_data_frame.norm_pos_x.values[gaze_index]
                gaze_norm_y = gaze_data_frame.norm_pos_y.values[gaze_index]

                gaze_pixel_x = int(gaze_norm_x * horizontal_pixels * scale_x)
                gaze_pixel_y = int((1 - gaze_norm_y) * vertical_pixels * scale_y)
                img_1 = cv2.circle(img_1, (gaze_pixel_x, gaze_pixel_y), 10, (255,255,0), 8)

                corners2 = np.squeeze(corners2)
                marker_position = np.mean(corners2, axis=0)
                img_1 = cv2.circle(img_1, (marker_position[0], marker_position[1]), 6, (255,255,255), 8)
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
                text = 'confidence: ' + str(np.round(gaze_data_frame.confidence.values[gaze_index], 3))
                # Using cv2.putText() method
                img = cv2.putText(img_1, text, org, font,
                                    font_scale, color, thickness, cv2.LINE_AA)

                # print('before : ', corners)
                corners2[:, 0] = corners2[:, 0] * (1 / scale_x)
                corners2[:, 1] = corners2[:, 1] * (1 / scale_y)
                # print('after : ', corners)

                imgpoints.append(corners2)
                image_list.append(img_1)
                marker_found_index.append(count)
                my_string = '1'
            else:
                my_string = '0'
            cv2.imshow('img', img)
            if cv2.waitKey(2) & 0xFF == ord('q'):
                break
    print('\nDone!')
    cv2.destroyAllWindows()

    return imgpoints, image_list, marker_found_index


if __name__ == "__main__":

    # dataPath = '/hdd01/kamran_sync/vedb/recordings_pilot/pilot_study_1/423/'
    dataPath = "/home/veddy06/kamran_sync/vedb/recordings_pilot/pilot_study_1/423/"
    subFolder = '/000/'

    fps = 30
    safeMargin = 5
    # start and end of the search window
    start = 0 * 60 + 55
    end = 1 * 60 + 45
    # number of rows and columns on the checkerboard
    checker_board_size = (8, 6)
    # to speed up the process
    scale_factor = (0.5, 0.5)

    points, images, marker_index = detect_checkerboard(dataPath, checker_board_size, scale_factor, start, end)