import os
import sys
import glob
import yaml
from data_analysis import gaze

# Directory for the recording sessions
fdir = "/hdd01/kamran_sync/vedbcloud0/"
next_in_line = ["2020_08_23_19_43_29", "000", "2020_08_23_20_05_26", "2020_08_23_22_27_12", "2020_09_14_13_54_11",
                "2020_10_15_12_30_26", "2020_10_15_12_05_24", "2020_10_19_00_23_14", "2020_10_16_11_07_40",
                "2020_10_16_11_26_00"]
#next_in_line = ["2020_08_23_19_43_29"]
#done = ["2020_10_15_12_05_24", "2020_10_19_00_23_14", "2020_10_16_11_07_40", "2020_10_16_11_26_00"]
parameters_fpath = "/home/kamran/Code/data_analysis/data_analysis/gaze/gaze_calibration_parameters.yaml"
def parse_pipeline_parameters(parameters_fpath):
    param_dict = dict()
    with open(parameters_fpath,"r") as stream:
        param_dict = yaml.safe_load(stream)
    return param_dict

if __name__ == "__main__":
    param_dict = parse_pipeline_parameters(parameters_fpath)
    session_directory = param_dict['directory']['session_directory']
#    sessions = glob.glob(session_directory + "*")

    sessions = ["/hdd01/kamran_sync/vedbcloud0/staging/2020_10_19_00_23_14",
               "/hdd01/kamran_sync/vedbcloud0/staging/2020_10_15_12_05_24",
                "/hdd01/kamran_sync/vedbcloud0/staging/2020_10_15_12_30_26",
                "/hdd01/kamran_sync/vedbcloud0/staging/2020_10_16_11_07_40",
                "/hdd01/kamran_sync/vedbcloud0/staging/2020_11_03_02_27_22",
                "/hdd01/kamran_sync/vedbcloud0/staging/2020_12_08_06_39_12",
                "/hdd01/kamran_sync/vedbcloud0/staging/2020_12_12_03_14_20",
                "/hdd01/kamran_sync/vedbcloud0/staging/2020_12_13_11_12_10",
                "/hdd01/kamran_sync/vedbcloud0/staging/2020_12_18_12_04_33",
                "/hdd01/kamran_sync/vedbcloud0/staging/2021_01_11_16_33_39",
                "/hdd01/kamran_sync/vedbcloud0/staging/2021_01_18_16_56_46",
                "/hdd01/kamran_sync/vedbcloud0/staging/2021_01_20_11_32_15",
                "/hdd01/kamran_sync/vedbcloud0/staging/2021_02_01_12_35_10",
                "/hdd01/kamran_sync/vedbcloud0/staging/2021_02_02_11_14_14",
                "/hdd01/kamran_sync/vedbcloud0/staging/2020_11_03_04_50_44",
                ]

                # "/hdd01/kamran_sync/vedbcloud0/staging/000",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2020_08_23_19_43_29",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2020_08_23_20_05_26",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2020_08_23_22_27_12",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2020_09_14_13_54_11",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2020_09_28_20_53_15",

    print("total number of sessions", len(sessions))
    # print("all sessions: ", sessions)
    for session_folder in sessions:
        result = False
        print("running analysis for:", session_folder)
        # if ("2020_10_15_12_05_24" in session_folder): #2020_10_15_12_05_24  #2020_10_19_00_23_14
        try:
            result = gaze.pipelines.pupil_2d_binocular_v01(
                    session_folder, param_dict, batch_size_pupil=500, batch_size_marker=100)
        except:
           print("Failed for session %s " % session_folder)
        # else:
        #     print("skipped session: ", session_folder)

        print("Gaze Calibratrion Result: ", result)
