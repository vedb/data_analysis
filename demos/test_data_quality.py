import pupil_recording_interface as pri
import cv2
import numpy as np
import random
from pupil_detectors import Detector2D, Detector3D
import os
import pandas as pd
from tqdm import tqdm
from pupil_recording_interface.externals.circle_detector import find_pupil_circle_marker


class session:
    """ A light Session class to work with different streams of data in a recording session """

    def __init__(self, path):
        self.session_path = path
        print("Session file Created for: {}".format(os.path.basename(os.path.dirname(path))))
        self.world_video_file = os.path.join(path, "world.mp4")
        self.eye0_video_file = os.path.join(path, "eye0.mp4")
        self.eye1_video_file = os.path.join(path, "eye1.mp4")
        self.t265_video_file = os.path.join(path, "t265.mp4")

        self.world_timestamp_file = os.path.join(path, "world_timestamps.npy")
        self.eye0_timestamp_file = os.path.join(path, "eye0_timestamps.npy")
        self.eye1_timestamp_file = os.path.join(path, "eye1_timestamps.npy")
        self.gyro_timestamp_file = os.path.join(path, "gyro_timestamps.npy")
        self.accel_timestamp_file = os.path.join(path, "accel_timestamps.npy")
        self.odometry_timestamp_file = os.path.join(path, "odometry_timestamps.npy")
        self.t265_timestamp_file = os.path.join(path, "t265_timestamps.npy")

        self.world_video_isOpen = False
        self.eye0_video_isOpen = False
        self.eye1_video_isOpen = False
        self.t265_video_isOpen = False

        self.world_video = None
        self.eye0_video = None
        self.eye1_video = None
        self.t265_video = None

        self.world_timestamp = None
        self.eye0_timestamp = None
        self.eye1_timestamp = None
        self.gyro_timestamp = None
        self.accel_timestamp = None
        self.odometry_timestamp = None
        self.t265_timestamp = None

    def open_video_cv2(self, video_name):
        if "world" in video_name:
            if self.world_video_isOpen == False:
                self.world_video = cv2.VideoCapture(self.world_video_file)
                self.world_video_isOpen = True
            return self.world_video
        if "eye0" in video_name:
            if self.eye0_video_isOpen == False:
                self.eye0_video = cv2.VideoCapture(self.eye0_video_file)
                self.eye0_video_isOpen = True
            return self.eye0_video
        if "eye1" in video_name:
            if self.eye1_video_isOpen == False:
                self.eye1_video = cv2.VideoCapture(self.eye1_video_file)
                self.eye1_video_isOpen = True
            return self.eye1_video
        if "t265" in video_name:
            if self.t265_video_isOpen == False:
                self.t265_video = cv2.VideoCapture(self.t265_video_file)
                self.t265_video_isOpen = True
            return self.t265_video

    def video_frame_size_cv2(self, video):
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        return (frame_width, frame_height)

    def close_video_cv2(self, video_name):
        if "world" in video_name:
            self.world_video.release()
        if "eye0" in video_name:
            self.eye0_video.release()
        if "eye1" in video_name:
            self.eye1_video.release()
        if "t265" in video_name:
            self.t265_video.release()

    def read_timestamp_np(self, array_name):
        if "world" in array_name:
            return np.load(self.world_timestamp_file)
        if "eye0" in array_name:
            return np.load(self.eye0_timestamp_file)
        if "eye1" in array_name:
            return np.load(self.eye1_timestamp_file)
        if "accel" in array_name:
            return np.load(self.accel_timestamp_file)
        if "gyro" in array_name:
            return np.load(self.gyro_timestamp_file)
        if "odometry" in array_name:
            return np.load(self.odometry_timestamp_file)
        if "t265" in array_name:
            return np.load(self.t265_timestamp_file)

    def read_all_timestamps(self):
        self.world_timestamp = self.read_timestamp_np("world")
        self.eye0_timestamp = self.read_timestamp_np("eye0")
        self.eye1_timestamp = self.read_timestamp_np("eye1")
        self.gyro_timestamp = self.read_timestamp_np("gyro")
        self.accel_timestamp = self.read_timestamp_np("accel")
        self.odometry_timestamp = self.read_timestamp_np("odometry")
        self.t265_timestamp = self.read_timestamp_np("t265")


def run_pupil_detection_PL(session, number_of_frames=250):
    pupil_dict_keys = ["pupil_timestamp", "eye_index", "world_index", "eye_id", "confidence", "norm_pos_x",
                       "norm_pos_y",
                       "diameter", "method", "ellipse_center_x", "ellipse_center_y", "ellipse_axis_a", "ellipse_axis_b",
                       "ellipse_angle", "diameter_3d", "model_confidence", "model_id", "sphere_center_x",
                       "sphere_center_y", "sphere_center_z", "sphere_radius", "circle_3d_center_x",
                       "circle_3d_center_y", "circle_3d_center_z", "circle_3d_normal_x", "circle_3d_normal_y",
                       "circle_3d_normal_z", "circle_3d_radius", "theta", "phi", "projected_sphere_center_x",
                       "projected_sphere_center_y", "projected_sphere_axis_a", "projected_sphere_axis_b",
                       "projected_sphere_angle"]

    eye0_video = session.open_video_cv2(video_name="eye0")
    eye1_video = session.open_video_cv2(video_name="eye1")

    eye0_frame_size = session.video_frame_size_cv2(eye0_video)
    eye1_frame_size = session.video_frame_size_cv2(eye1_video)

    eye0_timestamp = session.read_timestamp_np("eye0")
    eye1_timestamp = session.read_timestamp_np("eye1")

    properties = {}
    detector_2D = Detector2D(properties=properties)
    detector_3D = Detector3D(properties=properties)
    detection_method = "2d c++"

    max_frame_number = min(int(eye0_video.get(cv2.CAP_PROP_FRAME_COUNT)) - 10,
                           int(eye1_video.get(cv2.CAP_PROP_FRAME_COUNT)) - 10)
    number_of_rows = 2 * max_frame_number
    df = pd.DataFrame(columns=pupil_dict_keys, index=np.arange(2 * number_of_frames), dtype=np.float64)
    eye_image_indexes = random.sample(range(10, max_frame_number), number_of_frames)
    print("Running pupil tracking for {} eye video frames".format(len(eye_image_indexes)))
    count = 0
    for frame_index in tqdm(eye_image_indexes):
        eye0_video.set(1, frame_index)
        eye1_video.set(1, frame_index)
        ret_0, img_0 = eye0_video.read()
        ret_1, img_1 = eye1_video.read()
        if not ret_0 or not ret_1:
            break
        if np.ndim(img_0) == 3:
            img_0 = img_0[:, :, 0]
        if np.ndim(img_1) == 3:
            img_1 = img_1[:, :, 0]
        pupil0_dict_2d = detector_2D.detect(np.ascontiguousarray(img_0))
        df["pupil_timestamp"].loc[count] = eye0_timestamp[frame_index]
        df["eye_index"].loc[count] = int(frame_index)
        df["eye_id"].loc[count] = 0
        df["confidence"].loc[count] = pupil0_dict_2d["confidence"]
        df["norm_pos_x"].loc[count] = pupil0_dict_2d["location"][0] / eye0_frame_size[0]
        df["norm_pos_y"].loc[count] = pupil0_dict_2d["location"][1] / eye0_frame_size[1]
        df["diameter"].loc[count] = pupil0_dict_2d["diameter"]
        df["method"].loc[count] = detection_method
        df["ellipse_center_x"].loc[count] = pupil0_dict_2d["ellipse"]["center"][0]
        df["ellipse_center_y"].loc[count] = pupil0_dict_2d["ellipse"]["center"][1]
        df["ellipse_axis_a"].loc[count] = pupil0_dict_2d["ellipse"]["axes"][0]
        df["ellipse_axis_b"].loc[count] = pupil0_dict_2d["ellipse"]["axes"][1]
        df["ellipse_angle"].loc[count] = pupil0_dict_2d["ellipse"]["angle"]
        count = count + 1

        pupil1_dict_2d = detector_2D.detect(np.ascontiguousarray(img_1))
        df["pupil_timestamp"].loc[count] = eye1_timestamp[frame_index]
        df["eye_index"].loc[count] = int(frame_index)
        df["eye_id"].loc[count] = 1
        df["confidence"].loc[count] = pupil1_dict_2d["confidence"]
        df["norm_pos_x"].loc[count] = pupil1_dict_2d["location"][0] / eye0_frame_size[0]
        df["norm_pos_y"].loc[count] = pupil1_dict_2d["location"][1] / eye0_frame_size[1]
        df["diameter"].loc[count] = pupil1_dict_2d["diameter"]
        df["method"].loc[count] = detection_method
        df["ellipse_center_x"].loc[count] = pupil1_dict_2d["ellipse"]["center"][0]
        df["ellipse_center_y"].loc[count] = pupil1_dict_2d["ellipse"]["center"][1]
        df["ellipse_axis_a"].loc[count] = pupil1_dict_2d["ellipse"]["axes"][0]
        df["ellipse_axis_b"].loc[count] = pupil1_dict_2d["ellipse"]["axes"][1]
        df["ellipse_angle"].loc[count] = pupil1_dict_2d["ellipse"]["angle"]
        count = count + 1

    session.close_video_cv2("eye0")
    session.close_video_cv2("eye1")
    cv2.destroyAllWindows()
    print("\nDone!")
    return df


def run_marker_detection_0(session, frame_range, skip_frame=4, scale=0.5):
    """Use opencv to detect calibration and validation patterns"""

    world_dict_keys = ["world_timestamp", "world_index", "eye_index", "method",
                       "cal_location_x", "cal_location_y",
                       "val_location_x", "val_location_y",
                       "cal_norm_pos_x", "cal_norm_pos_y",
                       "val_norm_pos_x", "val_norm_pos_y",
                       "cal_size"]
    world_video = session.open_video_cv2(video_name="world")
    world_frame_size = session.video_frame_size_cv2(world_video)
    world_timestamp = session.read_timestamp_np("world")

    properties = {}
    detection_method = "2d-pupil-labs"

    max_frame_number = int(world_video.get(cv2.CAP_PROP_FRAME_COUNT)) - 10
    world_image_indexes = np.arange(frame_range[0], frame_range[1], skip_frame)
    df = pd.DataFrame(columns=world_dict_keys, index=np.arange(len(world_image_indexes)), dtype=np.float64)
    print("Running calibration marker detection for {} world video frames".format(len(world_image_indexes)))
    count = 0
    t = tqdm(world_image_indexes, desc="Frame Count", leave=True)
    for frame_index in t:

        world_video.set(1, frame_index)
        ret, img = world_video.read()
        if not ret:
            break
        if scale is not None:
            img = cv2.resize(img, None, fx=scale, fy=scale)

        frame = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        vdim, hdim = frame.shape[:2]

        world_circles = find_pupil_circle_marker(frame, scale)
        for iw, w in enumerate(world_circles):
            ellipses = w["ellipses"]
            for e in ellipses:
                this_circle = {}
                ellipse_centers = np.array([e[0] for e in ellipses])
                df["cal_location_x"].loc[count] = ellipse_centers.mean(0).tolist()[0]
                df["cal_location_y"].loc[count] = ellipse_centers.mean(0).tolist()[1]
                df["cal_norm_pos_x"].loc[count] = ellipse_centers.mean(0).tolist()[0] / hdim
                df["cal_norm_pos_y"].loc[count] = ellipse_centers.mean(0).tolist()[1] / vdim

                ellipse_radii = np.array([e[1] for e in ellipses])
                df["cal_size"].loc[count] = ellipse_radii.max(0)[0]
                df["world_timestamp"].loc[count] = world_timestamp[frame_index]

                ref_pixel_x = int(df["cal_location_x"].loc[count])
                ref_pixel_y = int(df["cal_location_y"].loc[count])
                img = cv2.circle(img, (ref_pixel_x, ref_pixel_y), 8, (200, 25, 205), 2)
                frame = img
                count = count + 1
            count = count + 1

        t.set_description("Frame Count")
        t.refresh()  # to show immediately the update
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # # print('\nDone!')
    # cv2.destroyAllWindows()
    # # print("timestamp shape: ", np.asarray(times).shape)
    # print("locations shape: ", np.asarray(locations).shape)
    # print("mean locations shape: ", np.asarray(mean_locations).shape)
    # print("norm pos shape: ", np.asarray(norm_pos).shape)
    # print("mean norm pos shape: ", np.asarray(mean_norm_pos).shape)
    # print("loc: ", mean_locations)
    # print("norm: ", mean_norm_pos)
    session.close_video_cv2("world")
    cv2.destroyAllWindows()
    print("\nDone!")
    return df


def run_marker_detection_1(session, frame_range=(1, 2), skip_frame=4, scale=0.5):
    """Use opencv to detect calibration and validation patterns"""

    checkerboard_size = (6, 8)
    rows, cols = checkerboard_size
    # termination criteria: Make inputs?
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    world_dict_keys = ["world_timestamp", "world_index", "eye_index", "method",
                       "cal_location_x", "cal_location_y",
                       "val_location_x", "val_location_y",
                       "cal_norm_pos_x", "cal_norm_pos_y",
                       "val_norm_pos_x", "val_norm_pos_y"]
    world_video = session.open_video_cv2(video_name="world")
    world_frame_size = session.video_frame_size_cv2(world_video)
    world_timestamp = session.read_timestamp_np("world")

    properties = {}
    detection_method = "2d-opencv-v0"

    max_frame_number = int(world_video.get(cv2.CAP_PROP_FRAME_COUNT)) - 10
    world_image_indexes = np.arange(frame_range[0], frame_range[1], skip_frame)
    df = pd.DataFrame(columns=world_dict_keys, index=np.arange(len(world_image_indexes)), dtype=np.float64)
    print("Running validation marker detection for {} world video frames".format(len(world_image_indexes)))
    count = 0
    t = tqdm(world_image_indexes, desc="Frame Count", leave=True)
    for frame_index in t:

        world_video.set(1, frame_index)
        ret, img = world_video.read()
        if not ret:
            break
        if scale is not None:
            img = cv2.resize(img, None, fx=scale, fy=scale)

        frame = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        vdim, hdim = frame.shape[:2]

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(frame, checkerboard_size, None)

        # If found, add object points, image points (after refining them)
        if ret:
            corners2 = cv2.cornerSubPix(frame, corners, (11, 11), (-1, -1), criteria)
            corners = np.squeeze(corners2)
            # Draw and display the corners
            frame = cv2.drawChessboardCorners(img, (6, 8), corners, ret)
            corners[:, 0] = corners[:, 0] * (1 / scale)
            corners[:, 1] = corners[:, 1] * (1 / scale)
            marker_position = np.mean(corners, axis=0)
            df["world_timestamp"].loc[count] = world_timestamp[frame_index]
            df["world_index"].loc[count] = frame_index
            df["method"].loc[count] = detection_method
            df["val_location_x"].loc[count] = marker_position[0]
            df["val_location_y"].loc[count] = marker_position[1]
            corners[:, 0] = corners[:, 0] * (scale / hdim)
            corners[:, 1] = corners[:, 1] * (scale / vdim)
            marker_position = np.mean(corners, axis=0)
            df["val_norm_pos_x"].loc[count] = marker_position[0]
            df["val_norm_pos_y"].loc[count] = marker_position[1]
            count = count + 1

        t.set_description("Frame Count (Found %i)" % ret)
        t.refresh()  # to show immediately the update
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # # print('\nDone!')
    # cv2.destroyAllWindows()
    # # print("timestamp shape: ", np.asarray(times).shape)
    # print("locations shape: ", np.asarray(locations).shape)
    # print("mean locations shape: ", np.asarray(mean_locations).shape)
    # print("norm pos shape: ", np.asarray(norm_pos).shape)
    # print("mean norm pos shape: ", np.asarray(mean_norm_pos).shape)
    # print("loc: ", mean_locations)
    # print("norm: ", mean_norm_pos)
    session.close_video_cv2("world")
    cv2.destroyAllWindows()
    print("\nDone!")
    return df

if __name__ == "__main__":

    session_path = "/hdd01/kamran_sync/staging/2021_09_28_17_25_20/"
    my_session = session(session_path)
    N = 50
    pupil_dataframe = run_pupil_detection_PL(session=my_session, number_of_frames=N)

    print("\n\n Running Circle Detection ...\n\n")
    start_index = 60*30*int(input("Please enter start time(minutes):"))
    end_index = 60*30*int(input("Please enter end time(minutes):"))
    scale = 0.5
    world_dataframe_cal = run_marker_detection_0(session=my_session, frame_range=(start_index, end_index), skip_frame=8, scale=scale)


    print("\n\n Running Checkerboard Detection ...\n\n")
    start_index = 60*30*int(input("Please enter start time(minutes):"))
    end_index = 60*30*int(input("Please enter end time(minutes):"))
    scale = 0.5
    world_dataframe_val = run_marker_detection_1(session=my_session, frame_range=(start_index, end_index), skip_frame=8, scale=scale)
