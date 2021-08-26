import os
import sys
import glob
import yaml
from data_analysis import scene
# Directory for the recording sessions
parameters_fpath = "/home/kamran/Code/data_analysis/config/visualization_parameters.yaml"
sessions_fpath = "/home/kamran/Code/data_analysis/config/sessions_list.yaml"
def parse_pipeline_parameters(parameters_fpath):
    param_dict = dict()
    with open(parameters_fpath,"r") as stream:
        param_dict = yaml.safe_load(stream)
    return param_dict

if __name__ == "__main__":
    import numpy as np
    param_dict = parse_pipeline_parameters(parameters_fpath)
    sessions_dict = parse_pipeline_parameters(sessions_fpath)
    session_directory = param_dict['directory']['session_directory']
    gaze_directory = param_dict['directory']['gaze_directory']
    # sessions = glob.glob(gaze_directory + "*")
    sessions = sessions_dict['sessions']
    print("total number of sessions", len(sessions))
    print("all sessions: ", sessions)
    val_points = []
    val_gaze = []
    cal_points = []
    cal_gaze = []
    for session_id in sessions:
        session_folder = os.path.join(session_directory,session_id)
        result = False
        print("running analysis for:", session_folder)
        # try:
        result = scene.pipelines.detect_markers(
            session_directory, session_folder , param_dict)
        # except:
        #    print("Failed for session %s " % session_folder)
        print("Gaze Calibratrion Result: ", result)
        # imgpoints, image_list, marker_found_index
        # val_points.append(result[0])
        # val_gaze.append(result[1])
        # cal_points.append(result[2])
        # cal_gaze.append(result[3])

    # file_name = "/home/kamran/Desktop/gaze_accuracy.npz"
    # final_result = [np.asarray(val_points), np.asarray(val_gaze), np.asarray(cal_points), np.asarray(cal_gaze)]
    # np.savez(file_name, validation_points = final_result[0],  validation_gaze = final_result[1], calibration_points = final_result[2], calibration_gaze = final_result[3])
    print("Final Result File Saved")