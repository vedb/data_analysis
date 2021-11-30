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


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    final_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return final_img


def apply_preprocessing(img, gamma, beta):
    if gamma > 0:
        table = 255.0 * (np.linspace(0, 1, 256) ** gamma)
        contrast_image = cv2.LUT(np.array(img), table)
    else:
        contrast_image = img
    if beta != 0:
        new_image = increase_brightness(contrast_image.astype(np.uint8), value=beta)
    else:
        new_image = contrast_image.astype(np.uint8)

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # new_image = clahe.apply(img)
    # new_image = cv2.cvtColor(new_image, cv2.COLOR_GRAY2BGR)
    return new_image.astype(np.uint8)

def run_pupil_detection_PL(session, df, number_of_frames=500):

    eye0_video = session.open_video_cv2(video_name="eye0")
    eye1_video = session.open_video_cv2(video_name="eye1")

    eye0_frame_size = session.video_frame_size_cv2(eye0_video)
    eye1_frame_size = session.video_frame_size_cv2(eye1_video)

    eye0_timestamp = session.read_timestamp_np("eye0")
    eye1_timestamp = session.read_timestamp_np("eye1")

    properties = {"pupil_size_max": 400, "pupil_size_min": 10}
    detector_2D = Detector2D(properties=properties)
    detection_method = "2d c++"

    # print(df.index)
#     eye_image_indexes = random.sample(range(df.index.values[0], df.index.values[-1]), number_of_frames)
    print("Running pupil tracking for {} eye video frames".format(len(df.index.values)))
    count = df.index[0]
    # df_dict = df.to_dict('records')
    eye0_video.set(1, count)
    eye1_video.set(1, count)
    frame_index = 0
    my_list = list()
    df_keys = ["pupil_timestamp", "eye_index", "world_index", "eye_id", "confidence", "norm_pos_x",
                       "norm_pos_y",
                       "diameter", "method", "ellipse_center_x", "ellipse_center_y", "ellipse_axis_a", "ellipse_axis_b",
                       "ellipse_angle", "diameter_3d", "model_confidence", "model_id", "sphere_center_x",
                       "sphere_center_y", "sphere_center_z", "sphere_radius", "circle_3d_center_x",
                       "circle_3d_center_y", "circle_3d_center_z", "circle_3d_normal_x", "circle_3d_normal_y",
                       "circle_3d_normal_z", "circle_3d_radius", "theta", "phi", "projected_sphere_center_x",
                       "projected_sphere_center_y", "projected_sphere_axis_a", "projected_sphere_axis_b",
                       "projected_sphere_angle"]
    for rows in tqdm(df.index):
        ret_0, img_0 = eye0_video.read()
        ret_1, img_1 = eye1_video.read()

        if ret_0:
            row = dict(keys=df_keys)
            gamma = 2.2 #.4
            beta = 0 #40
            # img_0 = apply_preprocessing(img_0, gamma, beta)
            # clahe = cv2.createCLAHE(clipLimit=gamma, tileGridSize=(16, 16)) #1.5 and 8 x 8
            # img_0 = clahe.apply(np.array(np.uint8(img_0[:, :, 0])))
            pupil0_dict_2d = detector_2D.detect(np.ascontiguousarray(img_0[:, :, 0]))#
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
            # df_dict[count] = row
            my_list.append(row)
            count = count + 1
            # img_0 = cv2.cvtColor(img_0, cv2.COLOR_GRAY2BGR)
            # img_0 = cv2.ellipse(img_0, (pupil0_dict_2d["ellipse"]["center"],
            #                             pupil0_dict_2d["ellipse"]["axes"],
            #                             pupil0_dict_2d["ellipse"]["angle"]),
            #                     (255, 255, 0), 4)

        if ret_1:
            gamma = 0.2 # 0.1
            beta = 0 # -50
            row = dict(keys=df_keys)
            # img_1 = apply_preprocessing(img_1, gamma, beta)
            # clahe = cv2.createCLAHE(clipLimit=gamma, tileGridSize=(6, 6)) #1.5 and 8 x 8
            # img_1 = clahe.apply(np.array(np.uint8(img_1[:, :, 0])))
            pupil1_dict_2d = detector_2D.detect(np.ascontiguousarray(img_1[:, :, 0]))#
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
            # df_dict[count] = row
            my_list.append(row)
            count = count + 1
            # img_1 = cv2.cvtColor(img_1, cv2.COLOR_GRAY2BGR)
            # img_1 = cv2.ellipse(img_1, (pupil1_dict_2d["ellipse"]["center"],
            #                             pupil1_dict_2d["ellipse"]["axes"],
            #                             pupil1_dict_2d["ellipse"]["angle"]),
            #                     (255, 255, 0), 4)

        frame_index = frame_index + 1
        # cv2.imshow("Pupil Detection", np.concatenate((img_0, img_1), axis=1))
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

    # df = pd.DataFrame.from_records(df_dict)
    df = pd.DataFrame(my_list)
    df = df.drop('keys', axis=1)
    df = df.dropna(axis='index', how='all')
    df.to_pickle(session.gaze_saving_path + "/pupil_positions_PL.pkl")
    df.to_csv(session.gaze_saving_path + "/pupil_positions_PL.csv")
    session.close_video_cv2("eye0")
    session.close_video_cv2("eye1")
    cv2.destroyAllWindows()
    print("\nDone!")

    return df

'''
            "coarse_detection": True,
            "coarse_filter_min": 128,
            "coarse_filter_max": 280,
            "intensity_range": 23,
            "blur_size": 5,
            "canny_treshold": 160,
            "canny_ration": 2,
            "canny_aperture": 5,
            "pupil_size_max": 100,
            "pupil_size_min": 10,
            "strong_perimeter_ratio_range_min": 0.8,
            "strong_perimeter_ratio_range_max": 1.1,
            "strong_area_ratio_range_min": 0.6,
            "strong_area_ratio_range_max": 1.1,
            "contour_size_min": 5,
            "ellipse_roundness_ratio": 0.1,
            "initial_ellipse_fit_treshhold": 1.8,
            "final_perimeter_ratio_range_min": 0.6,
            "final_perimeter_ratio_range_max": 1.2,
            "ellipse_true_support_min_dist": 2.5,
            "support_pixel_ratio_exponent": 2.0,
'''