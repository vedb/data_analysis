from pupil_recording_interface.externals.circle_detector import find_pupil_circle_marker
import numpy as np
import cv2
import cv2.aruco as aruco
import  copy
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from data_analysis.utils import read_yaml
from data_analysis.utils import find_closest_timestamp_index

def ProcessMarkerDetection(target, name, arguments):
    return mp.Process(target=target, name=name, args=arguments)


def read_process_configs(pipeline_param_path, sessions_param_path, viz_param_path):
    pipeline_params = read_yaml(pipeline_param_path)
    session_params = read_yaml(sessions_param_path)
    viz_params = read_yaml(viz_param_path)
    return pipeline_params, session_params, viz_params


def create_output_data(df_index, marker_type="calibration", size=(6, 8)):

    if marker_type == "calibration":
        # Todo: come up with an idea to encode some sort of confidence value
        df_keys = ["world_timestamp", "eye0_index", "eye1_index", "world_index", "confidence",
                   "pos_x", "pos_y", "norm_pos_x", "norm_pos_y", "size_x", "size_y", "marker_type", "method"]
        df = pd.DataFrame(columns=df_keys, index=df_index, dtype=np.float64)
    elif marker_type == "validation":
        # Todo: come up with an idea to encode some sort of confidence value
        df_keys = ["world_timestamp", "eye0_index", "eye1_index", "world_index", "confidence",
                   "pos_x", "pos_y", "norm_pos_x", "norm_pos_y", "size_x", "size_y", "marker_type", "method"]
        for i in range(size[0] * size[1]):
            column_label_x = "p_" + str(i) + "_x"
            column_label_y = "p_" + str(i) + "_y"
            df_keys.append(column_label_x)
            df_keys.append(column_label_y)
        df = pd.DataFrame(columns=df_keys, index=df_index, dtype=np.float64)
    else:
        print("Undefined Marker Type: ", marker_type)
        df_keys = ["a", "b", "c"]
        df = pd.DataFrame(columns=df_keys, index=df_index, dtype=np.float64)
    return df


def run_marker_detection_ved(session, df, marker_type="calibration", scale=1.0, start_index=0, end_index=10):

    world_video = session.open_video_cv2(video_name="world")
    world_frame_size = session.video_frame_size_cv2(world_video)

    eye0_timestamp = session.read_timestamp_np("eye0")
    eye1_timestamp = session.read_timestamp_np("eye1")
    world_timestamp = session.read_timestamp_np("world")


    # print(df.index)
    #     eye_image_indexes = random.sample(range(df.index.values[0], df.index.values[-1]), number_of_frames)
    world_indexes = np.arange(start_index, end_index)
    print("Running {} marker detection for {} world video frames".format(marker_type, len(world_indexes)))
    count = 0
    world_video.set(1, world_indexes[0])

    # Todo: Clean this up and use it in a better way
    markerSize = 6
    totalMarkers = 250
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()

    for frame_index in tqdm(world_indexes):
        ret, image = world_video.read()
        if not ret:
            break

        if "calibration" in marker_type:
            detection_method = "PL"
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            world_circles = find_pupil_circle_marker(gray, scale=scale)
            output = np.array([np.nan, np.nan])
            for iw, w in enumerate(world_circles):
                ellipses = w["ellipses"]
                ellipse_centers = np.array([e[0] for e in ellipses])
                location = ellipse_centers.mean(0).tolist()
                df["marker_type"].loc[count] = "calibration"
                df["method"].loc[count] = detection_method
                df["pos_x"].loc[count] = location[0] / scale
                df["pos_y"].loc[count] = location[1] / scale
                df["norm_pos_x"].loc[count] = location[0] / (world_frame_size[0] * scale)
                df["norm_pos_y"].loc[count] = location[1] / (world_frame_size[1] * scale)

                ellipse_radii = np.array([e[1] for e in ellipses])
                radius = ellipse_radii.max(0)
                df["size_x"].loc[count] = radius[0]
                df["size_y"].loc[count] = radius[1]
                df["world_timestamp"].loc[count] = world_timestamp[frame_index]
                df["world_index"].loc[count] = frame_index
                df["eye0_index"].loc[count] = find_closest_timestamp_index(ts=world_timestamp[frame_index],
                                                                           timestamp_array=eye0_timestamp)
                df["eye1_index"].loc[count] = find_closest_timestamp_index(ts=world_timestamp[frame_index],
                                                                           timestamp_array=eye1_timestamp)
                # Todo: come up with an idea to encode some sort of confidence value
                df["confidence"].loc[count] = np.NaN
                count = count + 1
                temp_img = cv2.circle(cv2.resize(image, None, fx=scale, fy=scale),
                                      (int(location[0] * scale),
                                       int(location[1] * scale)),
                                      14, (255, 255, 0), 8,)
                cv2.imshow("Calibration Marker Detection", temp_img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        elif "validation" in marker_type:
            detection_method = "ved"
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # bboxs, ids, rejected = findArucoMarkers(gray, arucoDict, arucoParam)
            # if ids is not None:
            #     temp_img = aruco.drawDetectedMarkers(cv2.resize(image, None, fx=scale, fy=scale), bboxs, ids)
            # Todo: Make this a parameter from the dictionary
            checkerboard_size = (6, 8)
            ret, corners = find_checkerboard(gray, scale=scale, checkerboard_size=checkerboard_size)

            if ret:
                corners[:, 0] = corners[:, 0] * (1 / scale)
                corners[:, 1] = corners[:, 1] * (1 / scale)
                location = np.mean(corners, axis=0)
                df["marker_type"].loc[count] = "validation"
                df["method"].loc[count] = detection_method
                df["pos_x"].loc[count] = location[0]
                df["pos_y"].loc[count] = location[1]
                df["norm_pos_x"].loc[count] = location[0] / (world_frame_size[0])
                df["norm_pos_y"].loc[count] = location[1] / (world_frame_size[1])

                size_x = abs(np.max(corners[:, 0]) - np.min(corners[:, 0]))
                size_y = abs(np.max(corners[:, 1]) - np.min(corners[:, 1]))
                df["size_x"].loc[count] = size_x
                df["size_y"].loc[count] = size_y
                df["world_timestamp"].loc[count] = world_timestamp[frame_index]
                df["world_index"].loc[count] = frame_index
                df["eye0_index"].loc[count] = find_closest_timestamp_index(ts=world_timestamp[frame_index],
                                                                           timestamp_array=eye0_timestamp)
                df["eye1_index"].loc[count] = find_closest_timestamp_index(ts=world_timestamp[frame_index],
                                                                           timestamp_array=eye1_timestamp)
                # Todo: come up with an idea to encode some sort of confidence value
                df["confidence"].loc[count] = np.NaN

                for i in range(checkerboard_size[0] * checkerboard_size[1]):
                    column_label_x = "p_" + str(i) + "_x"
                    column_label_y = "p_" + str(i) + "_y"
                    df[column_label_x].loc[count] = corners[i, 0]
                    df[column_label_y].loc[count] = corners[i, 1]

                count = count + 1
                temp_img = cv2.drawChessboardCorners(cv2.resize(image, None, fx=scale, fy=scale), (6, 8), corners*scale, True)
                cv2.imshow("Validation Marker Detection", temp_img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    df = df.dropna(axis='index', how='all')
    df.to_pickle(session.gaze_saving_path + "/world_"+marker_type+"_marker.pkl")
    df.to_csv(session.gaze_saving_path + "/world_"+marker_type+"_marker.csv")
    session.close_video_cv2("world")
    cv2.destroyAllWindows()
    print("\nDone!")
    return df


def _opencv_ellipse_to_dict(ellipse_dict):
    data = {}
    data["ellipse"] = {
        "center": (ellipse_dict.ellipse.center[0], ellipse_dict.ellipse.center[1]),
        "axes": (
            ellipse_dict["ellipses"].minor_radius * 2.0,
            ellipse_dict.ellipse.major_radius * 2.0,
        ),
        "angle": ellipse_dict.ellipse.angle * 180.0 / np.pi - 90.0,
    }
    data["diameter"] = max(data["ellipse"]["axes"])
    data["location"] = data["ellipse"]["center"]
    data["confidence"] = ellipse_dict.confidence


# def find_concentric_circles(image, scale=1.0):
#     """Assumes uint8 RGB image input"""
#     width, height, n_channels  = image.shape
#     width = width * scale
#     vdim = height * scale
#     out = []
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     world_circles = find_pupil_circle_marker(gray, scale)
#     return world_circles

def find_checkerboard(frame, scale=1.0, checkerboard_size=(6, 8)):
    # termination criteria: Make inputs?
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    if scale is not None:
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
        vdim, hdim = frame.shape[:2]
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(frame, checkerboard_size, None)
    # If found, add object points, image points (after refining them)
    if ret:
        corners2 = cv2.cornerSubPix(frame, corners, (11, 11), (-1, -1), criteria)
        corners = np.squeeze(corners2)
    return ret, corners

def findArucoMarkers(gray, arucoDict, arucoParam):

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bboxs, ids, rejected = aruco.detectMarkers(gray, arucoDict, parameters=arucoParam)

    return bboxs, ids, rejected
    # if (ids is not None):
    #     print("ID: ", ids)
    # if draw:
    #     aruco.drawDetectedMarkers(img, bboxs)

# def find_checkerboard(
#     video_data, timestamps=None, checkerboard_size=(6, 8), scale=None, progress_bar=None
# ):
#     """Use opencv to detect checkerboard pattern"""
#     if progress_bar is None:
#         progress_bar = lambda x: x
#
#     rows, cols = checkerboard_size
#     n_frames, vdim, hdim = video_data.shape[:3]
#     times = []  # frame timestamps for detected keypoints
#     locations = []  # 2d points in image plane.
#     norm_pos = []  # 2d normalized points in image plane.
#     mean_locations = []  # Mean 2d points in image plane
#     mean_norm_pos = []  # Mean 2d normalized points in image plane
#     reference_dict = {}
#     # termination criteria: Make inputs?
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#
#     n_iter = min([len(timestamps), len(video_data)])
#     for frame_time, frame in progress_bar(zip(timestamps, video_data), total=n_iter):
#         if np.ndim(video_data) == 4:
#             # Color image; remove color
#             # TODO: add option for BGR image?
#             # color_frame = copy.deepcopy(cv2.resize(frame, None, fx=scale, fy=scale))
#             # color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
#             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#         if scale is not None:
#             frame = cv2.resize(frame, None, fx=scale, fy=scale)
#             vdim, hdim = frame.shape[:2]
#         # Find the chess board corners
#         ret, corners = cv2.findChessboardCorners(frame, checkerboard_size, None)
#         # If found, add object points, image points (after refining them)
#         if ret:
#             corners2 = cv2.cornerSubPix(frame, corners, (11, 11), (-1, -1), criteria)
#             times.append(frame_time)
#             corners = np.squeeze(corners2)
#             # Draw and display the corners
#             # frame = cv2.drawChessboardCorners(color_frame, (6, 8), corners, ret)
#             corners[:, 0] = corners[:, 0] * (1 / scale)
#             corners[:, 1] = corners[:, 1] * (1 / scale)
#             locations.append(corners)
#             marker_position = np.mean(corners, axis=0)
#             mean_locations.append(marker_position)
#             corners[:, 0] = corners[:, 0] * (scale/hdim)
#             corners[:, 1] = corners[:, 1] * (scale/vdim)
#             norm_pos.append(corners)
#             marker_position = np.mean(corners, axis=0)
#             mean_norm_pos.append(marker_position)
#
#     #     cv2.imshow('Frame', frame)
#     #     if cv2.waitKey(1) & 0xFF == ord('q'):
#     #        break
#     # # print('\nDone!')
#     # cv2.destroyAllWindows()
#     # # print("timestamp shape: ", np.asarray(times).shape)
#     # print("locations shape: ", np.asarray(locations).shape)
#     # print("mean locations shape: ", np.asarray(mean_locations).shape)
#     # print("norm pos shape: ", np.asarray(norm_pos).shape)
#     # print("mean norm pos shape: ", np.asarray(mean_norm_pos).shape)
#     # print("loc: ", mean_locations)
#     # print("norm: ", mean_norm_pos)
#     reference_dict['location'] = np.asarray(locations)
#     reference_dict['norm_pos'] = np.asarray(norm_pos)
#     reference_dict['mean_location'] = np.asarray(mean_locations)
#     reference_dict['mean_norm_pos'] = np.asarray(mean_norm_pos)
#     reference_dict['timestamp'] = np.asarray(times)
#     return [reference_dict]#(np.asarray(x) for x in [times, locations, mean_locations, norm_pos, mean_norm_pos])


def find_checkerboard_v2(
    video_object, timestamps=None, checkerboard_size=(6, 8), scale=None, progress_bar=None
):
    """Use opencv to detect checkerboard pattern"""
    if progress_bar is None:
        progress_bar = lambda x: x

    rows, cols = checkerboard_size
    n_frames, vdim, hdim = video_data.shape[:3]
    times = []  # frame timestamps for detected keypoints
    locations = []  # 2d points in image plane.
    norm_pos = []  # 2d normalized points in image plane.
    mean_locations = []  # Mean 2d points in image plane
    mean_norm_pos = []  # Mean 2d normalized points in image plane
    reference_dict = {}
    # termination criteria: Make inputs?
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    n_iter = min([len(timestamps), len(video_data)])
    for frame_time, frame in progress_bar(zip(timestamps, video_data), total=n_iter):
        if np.ndim(video_data) == 4:
            # Color image; remove color
            # TODO: add option for BGR image?
            # color_frame = copy.deepcopy(cv2.resize(frame, None, fx=scale, fy=scale))
            # color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if scale is not None:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
            vdim, hdim = frame.shape[:2]
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(frame, checkerboard_size, None)
        # If found, add object points, image points (after refining them)
        if ret:
            corners2 = cv2.cornerSubPix(frame, corners, (11, 11), (-1, -1), criteria)
            times.append(frame_time)
            corners = np.squeeze(corners2)
            # Draw and display the corners
            # frame = cv2.drawChessboardCorners(color_frame, (6, 8), corners, ret)
            corners[:, 0] = corners[:, 0] * (1 / scale)
            corners[:, 1] = corners[:, 1] * (1 / scale)
            locations.append(corners)
            marker_position = np.mean(corners, axis=0)
            mean_locations.append(marker_position)
            corners[:, 0] = corners[:, 0] * (scale/hdim)
            corners[:, 1] = corners[:, 1] * (scale/vdim)
            norm_pos.append(corners)
            marker_position = np.mean(corners, axis=0)
            mean_norm_pos.append(marker_position)

    #     cv2.imshow('Frame', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #        break
    # # print('\nDone!')
    # cv2.destroyAllWindows()
    # # print("timestamp shape: ", np.asarray(times).shape)
    # print("locations shape: ", np.asarray(locations).shape)
    # print("mean locations shape: ", np.asarray(mean_locations).shape)
    # print("norm pos shape: ", np.asarray(norm_pos).shape)
    # print("mean norm pos shape: ", np.asarray(mean_norm_pos).shape)
    # print("loc: ", mean_locations)
    # print("norm: ", mean_norm_pos)
    reference_dict['location'] = np.asarray(locations)
    reference_dict['norm_pos'] = np.asarray(norm_pos)
    reference_dict['mean_location'] = np.asarray(mean_locations)
    reference_dict['mean_norm_pos'] = np.asarray(mean_norm_pos)
    reference_dict['timestamp'] = np.asarray(times)
    return [reference_dict]#(np.asarray(x) for x in [times, locations, mean_locations, norm_pos, mean_norm_pos])
