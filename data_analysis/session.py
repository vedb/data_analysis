import os
import numpy as np
import cv2

class Session:
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

    def video_total_frame_count_cv2(self, video):

        return int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    def close_video_cv2(self, video_name):
        if "world" in video_name:
            self.world_video_isOpen = False
            self.world_video.release()
        if "eye0" in video_name:
            self.eye0_video_isOpen = False
            self.eye0_video.release()
        if "eye1" in video_name:
            self.eye1_video_isOpen = False
            self.eye1_video.release()
        if "t265" in video_name:
            self.t265_video_isOpen = False
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
