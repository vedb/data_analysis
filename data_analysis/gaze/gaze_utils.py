# Utilities supporting gaze calibration

import numpy as np
import cv2
import pandas as pd
import os
from ..scene.scene_utils import undistort_unproject_pts

def onoff_from_binary(data, return_duration=True):
    """Converts a binary variable data into onsets, offsets, and optionally durations

    This may yield unexpected behavior if the first value of `data` is true.

    Parameters
    ----------
    data : array-like, 1D
        binary array from which onsets and offsets should be extracted

    """
    data = data.astype(np.float).copy()
    ddata = np.hstack([[0], np.diff(data)])
    (onsets,) = np.nonzero(ddata > 0)
    # print(onsets)
    (offsets,) = np.nonzero(ddata < 0)
    # print(offsets)
    onset_first = onsets[0] < offsets[0]
    len(onsets) == len(offsets)

    on_at_end = False
    on_at_start = False
    if onset_first:
        if len(onsets) > len(offsets):
            offsets = np.hstack([offsets, [-1]])
            on_at_end = True
    else:
        if len(offsets) > len(onsets):
            onsets = np.hstack([-1, offsets])
            on_at_start = True
    onoff = np.vstack([onsets, offsets])
    if return_duration:
        duration = offsets - onsets
        if on_at_end:
            duration[-1] = len(data) - onsets[-1]
        if on_at_start:
            duration[0] = offsets[0] - 0
        onoff = np.vstack([onoff, duration])

    onoff = onoff.T.astype(np.int)
    return onoff


def time_to_index(onsets_offsets, timeline):
    """find indices between onsets & offsets in timeline

    Parameters
    ----------
    """
    out = np.zeros_like(onsets_offsets)
    for ct, (on, off) in enumerate(onsets_offsets):
        i = np.flatnonzero(timeline > on)[0]
        j = np.flatnonzero(timeline < off)[-1]
        out[ct] = [i, j]
    return out


def dictlist_to_arraydict(dictlist):
    """Convert from pupil format list of dicts to dict of arrays"""
    dict_fields = list(dictlist[0].keys())
    out = {}
    for df in dict_fields:
        out[df] = np.array([d[df] for d in dictlist])
    return out


def arraydict_to_dictlist(arraydict):
    """Convert from dict of arrays to pupil format list of dicts"""
    dict_fields = list(arraydict.keys())
    first_key = dict_fields[0]
    n = len(arraydict[first_key])
    out = []
    for j in range(n):
        frame_dict = {}
        for k in dict_fields:
            value = arraydict[k][j]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            frame_dict[k] = value
        out.append(frame_dict)
    return out

def read_pl_gaze_csv(session_folder, output_id):
    sub_directory = str(output_id) * 3
    csv_file_name = os.path.join(session_folder,'exports',sub_directory,"gaze_positions.csv")
    print("CSV File Name: ", csv_file_name)
    return pd.read_csv(csv_file_name)

def find_focal_point_degrees(camera_matrix, dist_coefs):
    frame_width = 2048.0
    frame_height = 1536.0

    pts_3d_d = np.array([[frame_width], [frame_height]], dtype=np.float64).T
    #     print(type(np.float32(pts_3d_d)), pts_3d_d)

    pts_3d = undistort_unproject_pts(pts_3d_d, camera_matrix, dist_coefs)
    #     print("Undistorted Points_3d:\n", pts_3d)

    marker_pixel_x = np.arctan2(pts_3d[0], 1.0) * 180 / np.pi
    marker_pixel_y = np.arctan2(pts_3d[1], 1.0) * 180 / np.pi
    #     print(marker_pixel_x, marker_pixel_y)
    return marker_pixel_x, marker_pixel_y


def remove_non_fixation(m_x, m_y, g_x, g_y, error, threshold=2):
    #     threshold = 2
    #     print(len(g_x))
    print("Removing Non-fixation Eye Movements > {} degrees".format(threshold))
    x = np.diff(g_x, append=g_x[-1])
    y = np.diff(g_y, append=g_y[-1])
    D = np.power(x ** 2 + y ** 2, 0.5)
    #     print(len(D))

    out_m_x = m_x[abs(D) <= threshold]
    out_m_y = m_y[abs(D) <= threshold]
    out_g_x = g_x[abs(D) <= threshold]
    out_g_y = g_y[abs(D) <= threshold]
    error = error[abs(D) <= threshold]

    return out_m_x, out_m_y, out_g_x, out_g_y, error


def remove_outlier(m_x, m_y, g_x, g_y, error, threshold=10):
    print("Removing Outliers > {} degrees".format(threshold))
    out_m_x = m_x[error <= threshold]
    out_m_y = m_y[error <= threshold]
    out_g_x = g_x[error <= threshold]
    out_g_y = g_y[error <= threshold]
    error = error[error <= threshold]

    return out_m_x, out_m_y, out_g_x, out_g_y, error
