import numpy as np
import pandas as pd
import os
import cv2
from pupil_recording_interface.externals.gaze_mappers import Binocular_Gaze_Mapper
from .gaze_utils import onoff_from_binary
from pupil_recording_interface.externals.data_processing import (
    _filter_pupil_list_by_confidence,
    _extract_2d_data_monocular,
    _extract_2d_data_binocular,
    _match_data,
)
from pupil_recording_interface.externals import calibrate_2d


def get_data(pupil_list, ref_list, mode="2d", min_calibration_confidence=0.5):
    """Returns extracted data for calibration and whether there is binocular data

    Parameters
    ----------
    pupil_list :

    ref_list :
    """

    pupil_list = _filter_pupil_list_by_confidence(
        pupil_list, min_calibration_confidence
    )

    matched_data = _match_data(pupil_list, ref_list)
    (
        matched_binocular_data,
        matched_monocular_data,
        matched_pupil0_data,
        matched_pupil1_data,
    ) = matched_data

    binocular = None
    extracted_data = None
    if mode == "3d":
        if matched_binocular_data:
            binocular = True
            extracted_data = _extract_3d_data(g_pool, matched_binocular_data)
        elif matched_monocular_data:
            binocular = False
            extracted_data = _extract_3d_data(g_pool, matched_monocular_data)

    elif mode == "2d":
        if matched_binocular_data:
            binocular = True
            cal_pt_cloud_binocular = _extract_2d_data_binocular(matched_binocular_data)
            cal_pt_cloud0 = _extract_2d_data_monocular(matched_pupil0_data)
            cal_pt_cloud1 = _extract_2d_data_monocular(matched_pupil1_data)
            extracted_data = (
                cal_pt_cloud_binocular,
                cal_pt_cloud0,
                cal_pt_cloud1,
            )
        elif matched_monocular_data:
            binocular = False
            cal_pt_cloud = _extract_2d_data_monocular(matched_monocular_data)
            extracted_data = (cal_pt_cloud,)

    return binocular, extracted_data


def select_calibration_times(circle_positions, frame_times, stable_time=0.6):
    """"""
    # Find points with at least something detected
    circle_detected = np.all(~np.isnan(circle_positions), 1)
    on, off, dur = onoff_from_binary(circle_detected, return_duration=True).T

    durations = frame_times[off] - frame_times[on]
    # print(durations)
    keep = durations > stable_time
    out = np.vstack([frame_times[on[keep]], frame_times[off[keep]]]).T
    return out


def calibrate_2d_monocular(cal_pt_cloud, frame_size):
    method = "monocular polynomial regression"

    map_fn, inliers, params = calibrate_2d.calibrate_2d_polynomial(
        cal_pt_cloud, frame_size, binocular=False
    )
    if not inliers.any():
        return method, None

    mapper = "Monocular_Gaze_Mapper"
    args = {"params": params}
    result = {"subject": "start_plugin", "name": mapper, "args": args}
    return method, result


def calibrate_2d_binocular(cal_pt_cloud_binocular, cal_pt_cloud0,
                           cal_pt_cloud1, frame_size):
    method = "binocular polynomial regression"

    map_fn, inliers, params = calibrate_2d.calibrate_2d_polynomial(
        cal_pt_cloud_binocular, frame_size, binocular=True
    )
    if not inliers.any():
        return method, None

    map_fn, inliers, params_eye0 = calibrate_2d.calibrate_2d_polynomial(
        cal_pt_cloud0, frame_size, binocular=False
    )
    if not inliers.any():
        return method, None

    map_fn, inliers, params_eye1 = calibrate_2d.calibrate_2d_polynomial(
        cal_pt_cloud1, frame_size, binocular=False
    )
    if not inliers.any():
        return method, None

    mapper = "Binocular_Gaze_Mapper"
    args = {
        "params": params,
        "params_eye0": params_eye0,
        "params_eye1": params_eye1,
    }
    result = {"subject": "start_plugin", "name": mapper, "args": args}
    return method, result


def read_eye_calibration_data(path):
    try:
        pupil_df = pd.read_pickle(os.path.join(path, "pupil_positions_PL.pkl"))
    except:
        print("Falling back on to reading csv!!\n Try to update your pandas to version 1.3.0 for better performance!")
        pupil_df = pd.read_csv(os.path.join(path, "pupil_positions_PL.csv"))

    try:
        cal_marker_df = pd.read_pickle(os.path.join(path, "world_calibration_marker.pkl"))
    except:
        print("Falling back on to reading csv!!\n Try to update your pandas to version 1.3.0 for better performance!")
        cal_marker_df = pd.read_pickle(os.path.join(path, "world_calibration_marker.csv"))

    try:
        val_marker_df = pd.read_pickle(os.path.join(path, "world_validation_marker.pkl"))
    except:
        print("Falling back on to reading csv!!\n Try to update your pandas to version 1.3.0 for better performance!")
        val_marker_df = pd.read_pickle(os.path.join(path, "world_validation_marker.csv"))

    return pupil_df, cal_marker_df, val_marker_df


def get_right_pupil_DF(p_df):
    # Todo: Combine the right and left methods into one and pass 0,1 as arg
    print("\n\nconverting Right pupil DataFrame to Pupil labs format!!")
    r_df = p_df.groupby("eye_id").get_group(0.0)
    right_pupil_df = pd.DataFrame(columns=['ellipse', 'diameter', 'location',
                                           'confidence', 'luminance', 'norm_pos',
                                           'timestamp', 'id'], index=r_df.index)
    for i in r_df.index:
        right_pupil_df.loc[i, "norm_pos"] = tuple([r_df.loc[i, 'norm_pos_x'], r_df.loc[i, 'norm_pos_y']])
        right_pupil_df.loc[i, "location"] = tuple([r_df.loc[i, 'ellipse_center_x'], r_df.loc[i, 'ellipse_center_y']])
        right_pupil_df.loc[i, "confidence"] = r_df.loc[i, "confidence"]
        right_pupil_df.loc[i, "timestamp"] = r_df.loc[i, "pupil_timestamp"]
        right_pupil_df.loc[i, "id"] = r_df.loc[i, "eye_id"]

    return right_pupil_df


def get_left_pupil_DF(p_df):
    # Todo: Combine the right and left methods into one and pass 0,1 as arg
    print("\n\nconverting Left pupil DataFrame to Pupil labs format!!")
    l_df = p_df.groupby("eye_id").get_group(1.0)
    left_pupil_df = pd.DataFrame(columns=['ellipse', 'diameter', 'location',
                                          'confidence', 'luminance', 'norm_pos',
                                          'timestamp', 'id'], index=l_df.index)
    for i in l_df.index:
        left_pupil_df.loc[i, "norm_pos"] = tuple([l_df.loc[i, 'norm_pos_x'], l_df.loc[i, 'norm_pos_y']])
        left_pupil_df.loc[i, "location"] = tuple([l_df.loc[i, 'ellipse_center_x'], l_df.loc[i, 'ellipse_center_y']])
        left_pupil_df.loc[i, "confidence"] = l_df.loc[i, "confidence"]
        left_pupil_df.loc[i, "timestamp"] = l_df.loc[i, "pupil_timestamp"]
        left_pupil_df.loc[i, "id"] = l_df.loc[i, "eye_id"]
    return left_pupil_df


def get_ref_DF(cal_df):
    print("\n\nconverting Calibration Marker DataFrame to Pupil labs format!!")
    ref_df = pd.DataFrame(columns=['location', 'norm_pos', 'size', 'timestamp'],
                          index=cal_df.index)
    for i in cal_df.index:
        ref_df.loc[i, "norm_pos"] = tuple([cal_df.loc[i, 'norm_pos_x'], cal_df.loc[i, 'norm_pos_y']])
        ref_df.loc[i, "location"] = tuple([cal_df.loc[i, 'pos_x'], cal_df.loc[i, 'pos_y']])
        ref_df.loc[i, "size"] = tuple([cal_df.loc[i, 'size_x'], cal_df.loc[i, 'size_y']])
    ref_df.loc[:, "timestamp"] = cal_df.loc[:, "world_timestamp"]
    return ref_df


def run_calibration_PL(pupil_list_right, pupil_list_left, ref_list):
    pupil_list_binocular = []
    pupil_list_binocular.extend(pupil_list_left)
    pupil_list_binocular.extend(pupil_list_right)

    # Calibrate
    # Get data for pupil calibration
    print("\n\nGetting data for calibration")
    is_binocular, matched_data_binocular = get_data(pupil_list_binocular, ref_list, mode="2d")
    # Run calibration
    print("\n\nRunning 2d binocular calibration")
    # Todo: Pass the frame size properly
    method, result = calibrate_2d_binocular(*matched_data_binocular, frame_size=(2048, 1536))  # This is Confirmed
    return method, result, pupil_list_binocular


def run_gaze_mapper_PL(result, pupil_list_binocular):
    # (6) Map gaze to video coordinates
    # Mapper takes two inputs: normalized pupil x and y position
    print("\n\nRunning gaze mapper [binocular]")

    # Create mapper for gaze
    if (result):
        binocular_gaze_mapper = Binocular_Gaze_Mapper(result["args"]["params"],
                                                      result["args"]["params_eye0"],
                                                      result["args"]["params_eye1"])
        gaze_binocular = binocular_gaze_mapper.map_batch(pupil_list_binocular)
        # Transpose output so time is the first dimension
        # TODO: Make sure the format is consistent with the monocular  gaze
        # gaze_binocular = np.vstack(gaze_binocular).T

        #     gaze_file = string_name.format(step="gaze")
        #     np.savez(gaze_file, gaze_binocular=gaze_binocular)
        final_result = True
    else:
        print("\n\nGaze Mapping Failed!")
        final_result = False
    return final_result, gaze_binocular


def clean_up_gaze_df(gaze_binocular):
    # This is mainly because to convert the nested dicts in PL format into one layer dataframe
    new_list = []
    for i in range(len(gaze_binocular)):
        new_list.append(gaze_binocular[i]["base_data"][0])
    new_df = pd.DataFrame.from_records(new_list)
    new_df = new_df.rename(columns={"confidence": "pupil_confidence",
                                    "norm_pos": "pupil_norm_pos",
                                    "timestamp": "pupil_timestamp"})

    binocular_gaze_df = pd.DataFrame.from_records(gaze_binocular)
    binocular_gaze_df = binocular_gaze_df.drop(['base_data'], axis=1)

    base_keys = gaze_binocular[0]["base_data"][0].keys()

    #     binocular_gaze_df
    merged_df = pd.concat([binocular_gaze_df, new_df], axis=1)
    return merged_df


def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


    return np.squeeze(pts_3d)
