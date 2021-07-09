import os
import sys
import glob
import yaml
from data_analysis import gaze

parameters_path = "../config/gaze_calibration_parameters.yaml"
sessions_path = "../config/sessions_list.yaml"
def parse_pipeline_parameters(parameters_fpath):
    param_dict = dict()
    with open(parameters_fpath,"r") as stream:
        param_dict = yaml.safe_load(stream)
    return param_dict

if __name__ == "__main__":
    param_dict = parse_pipeline_parameters(parameters_path)
    session_directory = param_dict['directory']['session_directory']
    sessions_dict = parse_pipeline_parameters(sessions_path)
    gaze_directory = param_dict['directory']['gaze_directory']
    # sessions = glob.glob(gaze_directory + "*")
    sessions = sessions_dict['sessions']


    print("total number of sessions", len(sessions))
    # print("all sessions: ", sessions)
    for session_folder in sessions:
        result = False
        print("running analysis for:", session_folder)
        # if ("2020_10_15_12_05_24" in session_folder): #2020_10_15_12_05_24  #2020_10_19_00_23_14
        # try:
        result = gaze.pipelines.pupil_2d_binocular_v01(
                session_folder, param_dict, batch_size_pupil=500, batch_size_marker=100)
        # except:
        #    print("Failed for session %s " % session_folder)
        # else:
        #     print("skipped session: ", session_folder)

        print("Gaze Calibratrion Result: ", result)
