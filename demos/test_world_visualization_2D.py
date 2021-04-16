import os
import sys
import glob
import yaml
from data_analysis import gaze
from data_analysis import visualization
# Directory for the recording sessions
parameters_fpath = "/home/kamran/Code/data_analysis/data_analysis/visualization/visualization_parameters.yaml"
sessions_fpath = "/home/kamran/Code/data_analysis/data_analysis/visualization/sessions_list.yaml"
def parse_pipeline_parameters(parameters_fpath):
    param_dict = dict()
    with open(parameters_fpath,"r") as stream:
        param_dict = yaml.safe_load(stream)
    return param_dict

if __name__ == "__main__":
    param_dict = parse_pipeline_parameters(parameters_fpath)
    sessions_dict = parse_pipeline_parameters(sessions_fpath)
    session_directory = param_dict['directory']['session_directory']
    gaze_directory = param_dict['directory']['gaze_directory']
    # sessions = glob.glob(gaze_directory + "*")
    sessions = sessions_dict['sessions']
    print("total number of sessions", len(sessions))
    print("all sessions: ", sessions)
    for session_id in sessions:
        session_folder = os.path.join(session_directory,session_id)
        result = False
        print("running analysis for:", session_folder)
        # try:
        result = visualization.pipelines.show_world_v01(
            session_directory, session_folder , param_dict)
        # except:
        #    print("Failed for session %s " % session_folder)
        print("Gaze Calibratrion Result: ", result)
