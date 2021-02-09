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
    #sessions = glob.glob(gaze_directory + "*")
    sessions = ["/hdd01/kamran_sync/vedbcloud0/processed/gaze_pilot/2020_10_19_00_23_14",
                "/hdd01/kamran_sync/vedbcloud0/processed/gaze_pilot/2020_10_15_12_05_24",
                "/hdd01/kamran_sync/vedbcloud0/processed/gaze_pilot/2020_10_15_12_30_26",
                "/hdd01/kamran_sync/vedbcloud0/processed/gaze_pilot/2020_10_16_11_07_40",
                "/hdd01/kamran_sync/vedbcloud0/processed/gaze_pilot/2020_11_03_02_27_22",
                "/hdd01/kamran_sync/vedbcloud0/processed/gaze_pilot/2020_12_08_06_39_12",
                "/hdd01/kamran_sync/vedbcloud0/processed/gaze_pilot/2020_12_12_03_14_20",
                "/hdd01/kamran_sync/vedbcloud0/processed/gaze_pilot/2020_12_13_11_12_10",
                "/hdd01/kamran_sync/vedbcloud0/processed/gaze_pilot/2020_12_18_12_04_33",
                "/hdd01/kamran_sync/vedbcloud0/processed/gaze_pilot/2021_01_11_16_33_39",
                "/hdd01/kamran_sync/vedbcloud0/processed/gaze_pilot/2021_01_18_16_56_46",
                "/hdd01/kamran_sync/vedbcloud0/processed/gaze_pilot/2021_01_20_11_32_15",
                "/hdd01/kamran_sync/vedbcloud0/processed/gaze_pilot/2021_02_01_12_35_10",
                "/hdd01/kamran_sync/vedbcloud0/processed/gaze_pilot/2021_02_02_11_14_14",
                "/hdd01/kamran_sync/vedbcloud0/processed/gaze_pilot/2020_11_03_04_50_44",
                ]

                # "/hdd01/kamran_sync/vedbcloud0/processed/gaze_pilot/000",
                # "/hdd01/kamran_sync/vedbcloud0/processed/gaze_pilot/2020_08_23_19_43_29",
                # "/hdd01/kamran_sync/vedbcloud0/processed/gaze_pilot/2020_08_23_20_05_26",
                # "/hdd01/kamran_sync/vedbcloud0/processed/gaze_pilot/2020_08_23_22_27_12",
                # "/hdd01/kamran_sync/vedbcloud0/processed/gaze_pilot/2020_09_14_13_54_11",
                # "/hdd01/kamran_sync/vedbcloud0/processed/gaze_pilot/2020_09_28_20_53_15",

    print("total number of sessions", len(sessions))
    print("all sessions: ", sessions)
    for session_folder in sessions:
        result = False
        print("running analysis for:", session_folder)
        try:
        # if ("2020_10_19_00_23_14" in session_folder):
            result = visualization.pipelines.show_gaze_accuracy_v01(
                session_directory, session_folder , param_dict)
            # break
        except:
          print("Failed for session %s " % session_folder)
        #     print("Gaze Calibratrion Result: ", result)
        # else:
        #     print("skipped session: ", session_folder)
