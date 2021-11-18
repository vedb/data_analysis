import pupil_detectors
import numpy as np
from pupil_detectors import Detector2D
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

def create_output_data(df_index):
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
    detection_method = "2d c++"

    # print(df.index)
#     eye_image_indexes = random.sample(range(df.index.values[0], df.index.values[-1]), number_of_frames)
    print("Running pupil tracking for {} eye video frames".format(len(df.index.values)))
    count = df.index[0]
    df_dict = df.to_dict('records')
    eye0_video.set(1, count)
    eye1_video.set(1, count)
    frame_index = 0
    for row in tqdm(df_dict):

        ret_0, img_0 = eye0_video.read()
        ret_1, img_1 = eye1_video.read()

        if ret_0:
            pupil0_dict_2d = detector_2D.detect(np.ascontiguousarray(img_0[:, :, 0]))
            row["pupil_timestamp"] = eye0_timestamp[frame_index]
            row["eye_index"] = int(frame_index)
            row["eye_id"] = 0
            row["confidence"] = pupil0_dict_2d["confidence"]
            row["norm_pos_x"] = pupil0_dict_2d["location"][0] / eye0_frame_size[0]
            row["norm_pos_y"] = pupil0_dict_2d["location"][1] / eye0_frame_size[1]
            row["diameter"] = pupil0_dict_2d["diameter"]
            row["method"] = detection_method
            row["ellipse_center_x"] = pupil0_dict_2d["ellipse"]["center"][0]
            row["ellipse_center_y"] = pupil0_dict_2d["ellipse"]["center"][1]
            row["ellipse_axis_a"] = pupil0_dict_2d["ellipse"]["axes"][0]
            row["ellipse_axis_b"] = pupil0_dict_2d["ellipse"]["axes"][1]
            row["ellipse_angle"] = pupil0_dict_2d["ellipse"]["angle"]
            df_dict[count] = row
            count = count + 1

        if ret_1:
            pupil1_dict_2d = detector_2D.detect(np.ascontiguousarray(img_1[:, :, 0]))
            row["pupil_timestamp"] = eye1_timestamp[frame_index]
            row["eye_index"] = int(frame_index)
            row["eye_id"] = 1
            row["confidence"] = pupil1_dict_2d["confidence"]
            row["norm_pos_x"] = pupil1_dict_2d["location"][0] / eye1_frame_size[0]
            row["norm_pos_y"] = pupil1_dict_2d["location"][1] / eye1_frame_size[1]
            row["diameter"] = pupil1_dict_2d["diameter"]
            row["method"] = detection_method
            row["ellipse_center_x"] = pupil1_dict_2d["ellipse"]["center"][0]
            row["ellipse_center_y"] = pupil1_dict_2d["ellipse"]["center"][1]
            row["ellipse_axis_a"] = pupil1_dict_2d["ellipse"]["axes"][0]
            row["ellipse_axis_b"] = pupil1_dict_2d["ellipse"]["axes"][1]
            row["ellipse_angle"] = pupil1_dict_2d["ellipse"]["angle"]
            df_dict[count] = row
            count = count + 1
        frame_index = frame_index + 1
    df = pd.DataFrame.from_records(df_dict)
    df = df.dropna(axis='index', how='all')
    df.to_pickle(session.gaze_saving_path + "/pupil_positions_PL.pkl")
    df.to_csv(session.gaze_saving_path + "/pupil_positions_PL.csv")
    session.close_video_cv2("eye0")
    session.close_video_cv2("eye1")
    cv2.destroyAllWindows()
    print("\nDone!")

    return df
