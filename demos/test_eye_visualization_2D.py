import os
import sys
import glob
import yaml
from data_analysis import gaze
from data_analysis import visualization
# Directory for the recording sessions
parameters_fpath = "/home/kamran/Code/data_analysis/data_analysis/visualization/visualization_parameters.yaml"
def parse_pipeline_parameters(parameters_fpath):
    param_dict = dict()
    with open(parameters_fpath,"r") as stream:
        param_dict = yaml.safe_load(stream)
    return param_dict

if __name__ == "__main__":
    param_dict = parse_pipeline_parameters(parameters_fpath)
    session_directory = param_dict['directory']['session_directory']
    gaze_directory = param_dict['directory']['gaze_directory']
    sessions = glob.glob(gaze_directory + "*")
    print("total number of sessions", len(sessions))
    print("all sessions: ", sessions)
    for session_folder in sessions:
        result = False
        print("running analysis for:", session_folder)
        #try:
        if ("2020_10_19_00_23_14" in session_folder):
            result = visualization.pipelines.show_eye_v01(
                session_directory, session_folder , param_dict)
        #except:
        #   print("Failed for session %s " % session_folder)
            print("Gaze Calibratrion Result: ", result)
        else:
            print("skipped session: ", session_folder)
