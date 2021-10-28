import pupil_detectors
import numpy as np
from pupil_detectors import Detector2D, Detector3D
from data_analysis.process import BaseProcess
import cv2
import pandas as pd
import random
from tqdm import tqdm
import multiprocessing as mp
from data_analysis.utils import read_yaml


def ProcessPupilDetection(target, name, arguments):
    return mp.Process(target=target, name=name, args=arguments)


def read_process_configs(pipeline_param_path, sessions_param_path, viz_param_path):
    pipeline_params = read_yaml(pipeline_param_path)
    session_params = read_yaml(sessions_param_path)
    viz_params = read_yaml(viz_param_path)
    return pipeline_params, session_params, viz_params

def creat_output_data(df_index):
    df_keys = ["pupil_timestamp", "eye_index", "world_index", "eye_id", "confidence", "norm_pos_x",
                       "norm_pos_y",
                       "diameter", "method", "ellipse_center_x", "ellipse_center_y", "ellipse_axis_a", "ellipse_axis_b",
                       "ellipse_angle", "diameter_3d", "model_confidence", "model_id", "sphere_center_x",
                       "sphere_center_y", "sphere_center_z", "sphere_radius", "circle_3d_center_x",
                       "circle_3d_center_y", "circle_3d_center_z", "circle_3d_normal_x", "circle_3d_normal_y",
                       "circle_3d_normal_z", "circle_3d_radius", "theta", "phi", "projected_sphere_center_x",
                       "projected_sphere_center_y", "projected_sphere_axis_a", "projected_sphere_axis_b",
                       "projected_sphere_angle"]
    df = pd.DataFrame(columns=df_keys, index=df_index, dtype=np.float64)
    return df


def run_pupil_detection_PL(session, df, number_of_frames=500):

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

    # print(df.index)
#     eye_image_indexes = random.sample(range(df.index.values[0], df.index.values[-1]), number_of_frames)
    eye_image_indexes = df.index.values
    print("Running pupil tracking for {} eye video frames".format(len(eye_image_indexes)))
    count = df.index[0]
    eye0_video.set(1, count)
    eye1_video.set(1, count)
    for frame_index in tqdm(eye_image_indexes):
        ret_0, img_0 = eye0_video.read()
        ret_1, img_1 = eye1_video.read()
        if not ret_0 or not ret_1:
            break
        if np.ndim(img_0) == 3:
            img_0 = img_0[:, :, 0]
        if np.ndim(img_1) == 3:
            img_1 = img_1[:, :, 0]
        if count == 0:
            prev_frame_0 = np.array(img_0, dtype=np.float64)
            prev_frame_1 = np.array(img_1, dtype=np.float64)
        else:
            prev_frame_0 = (img_0 + prev_frame_0)/2.0
            prev_frame_1 = (img_1 + prev_frame_1)/2.0

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
    df.to_pickle("/home/kamran/Desktop/pupil_data.pkl")
    df.to_csv("/home/kamran/Desktop/pupil_data.csv")
    session.close_video_cv2("eye0")
    session.close_video_cv2("eye1")
    cv2.destroyAllWindows()
    print("\nDone!")
    return df.values.tolist()

