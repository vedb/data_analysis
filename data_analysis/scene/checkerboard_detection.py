import numpy as np
import cv2
import pandas as pd


def detect_checkerboard(path, checkerboard_size, scale, start_seconds, end_seconds):

    horizontal_pixels = 2048
    vertical_pixels = 1536

    rows = checkerboard_size[0]
    cols = checkerboard_size[1]
    scale_x = scale
    scale_y = scale
    sub_folder = "/000/"

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

    # gaze_data_frame = pd.read_csv(path + "/exports" + sub_folder + "gaze_positions.csv")
    # world_time_stamps = np.load(path + "/world_timestamps.npy")

    print("reading video: ", path + "/world.mp4")
    cap = cv2.VideoCapture(path + "/world.mp4")
    frame_numbers = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total Number of Frames: ", frame_numbers)

    # start_index = (start_seconds + safe_margin) * fps
    # end_index = (end_seconds - safe_margin) * fps
    start_index = start_seconds
    end_index = end_seconds

    print("First Frame = %d" % start_index)
    print("Last Frame = %d" % end_index)

    print("scale[x,y] = ", scale_x, scale_y)

    my_string = "-"
    marker_found = []
    cap.set(1, 1600)
    for count in range(0, frame_numbers):

        print(
            "Progress: {0:.1f}% {s}".format(count * 100 / frame_numbers, s=my_string),
            end="\r",
            flush=False,
        )
        if count < start_index:
            ret, frame = cap.read()
            continue
        elif count >= end_index:
            break
        else:

            # Read the next frame from the video. If you set frame 749 above then the code will return the last frame.
            ret, img = cap.read()
            if ret:
                # img = cv2.medianBlur(img, 3)
                kernel = np.ones((7, 7), np.uint8)
                img = cv2.erode(img, kernel, iterations=1)
                img[1100:1536,:, :] = [100, 100, 100]
                img = cv2.resize(img, None, fx=scale_x, fy=scale_y)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Find the chess board corners
                # ret, corners = cv2.findChessboardCorners(gray, (6, 8), None)
                ret = []


                gray = np.float32(gray)
                dst = cv2.cornerHarris(gray, 7, 7, 0.15)
                # result is dilated for marking the corners, not important
                dst = cv2.dilate(dst, None)
                # Threshold for an optimal value, it may vary depending on the image.
                img[dst > 0.2 * dst.max()] = [0, 0, 225]
                corners_x = np.where(img[:,:,2] == 225)[0]
                corners_y = np.where(img[:, :, 2] == 225)[1]
                if(len(corners_x)>4000 and max(corners_x)<1400 and max(corners_y)<1400) and np.std(corners_x)<100 and np.std(corners_y)<100:
                    # print('Yes', corners_x.shape, corners_y.shape)# , np.std(corners_x), np.std(corners_y)
                    print('Yes', [np.mean(corners_x*(1/scale)), np.mean(corners_y*(1/scale))])
                    img = cv2.circle(
                        img,
                        (int(np.mean(corners_y)), int(np.mean(corners_x))),
                        6,
                        (255, 255, 0),
                        8,
                    )
                    marker_found.append(count)
                    imgpoints.append([np.mean(corners_x*(1/scale)), np.mean(corners_y*(1/scale))])
                    my_string = "1"
                else:
                    # print('No', corners_x.shape, corners_y.shape)
                    # marker_found.append(False)
                    my_string = "0"

                cv2.imshow("world", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    print("\nDone!")
    cv2.destroyAllWindows()

    return imgpoints, marker_found


# if __name__ == "__main__":
#
#     # dataPath = '/hdd01/kamran_sync/vedb/recordings_pilot/pilot_study_1/423/'
#     dataPath = "/home/veddy06/kamran_sync/vedb/recordings_pilot/pilot_study_1/423/"
#     subFolder = "/000/"
#
#     fps = 30
#     safeMargin = 5
#     # start and end of the search window
#     start = 0 * 60 + 55
#     end = 1 * 60 + 45
#     # number of rows and columns on the checkerboard
#     checker_board_size = (8, 6)
#     # to speed up the process
#     scale_factor = (0.5, 0.5)
#
#     points, images, marker_index = detect_checkerboard(
#         dataPath, checker_board_size, scale_factor, start, end
#     )
