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

    sessions = [
                "/hdd01/kamran_sync/vedbcloud0/raw/2020_11_06_13_38_53",
                "/hdd01/kamran_sync/vedbcloud0/raw/2020_10_11_13_31_08",
                "/hdd01/kamran_sync/vedbcloud0/raw/2020_10_10_15_06_26",
                "/hdd01/kamran_sync/vedbcloud0/raw/2020_09_27_14_47_51",
                "/hdd01/kamran_sync/vedbcloud0/raw/2020_09_26_13_01_11",
                "/hdd01/kamran_sync/vedbcloud0/raw/2020_09_26_11_03_22",
                # "/hdd01/kamran_sync/vedbcloud0/raw/2020_09_23_16_55_17", # Corrupt World Video
                "/hdd01/kamran_sync/vedbcloud0/raw/2020_09_23_16_46_05",
                "/hdd01/kamran_sync/vedbcloud0/raw/2020_09_23_09_41_53",
                "/hdd01/kamran_sync/vedbcloud0/raw/2020_09_20_14_00_00",

                # "/hdd01/kamran_sync/vedbcloud0/staging/2021_03_04_16_16_45",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2021_03_09_11_33_49",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2021_03_14_14_13_55",

                # "/hdd01/kamran_sync/vedbcloud0/staging/2021_03_04_17_01_45",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2021_03_04_22_37_35",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2021_03_14_14_02_16",


                # "/hdd01/kamran_sync/vedbcloud0/staging/2021_02_23_14_14_57",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2021_02_25_16_28_15",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2021_02_04_18_36_36",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2021_02_08_16_24_36",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2021_02_11_18_00_52",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2021_02_19_13_53_29"
                # "/hdd01/kamran_sync/vedbcloud0/staging/2021_02_21_09_56_26", # Mark's outdoor data, washed out eye
                # "/hdd01/kamran_sync/vedbcloud0/staging/2021_01_21_17_14_05",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2021_01_21_20_58_27",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2021_02_12_13_17_27",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2021_01_28_12_18_46",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2021_01_14_15_00_12",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2021_01_12_12_59_50",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2021_01_09_10_44_23",

                # "/hdd01/kamran_sync/vedbcloud0/staging/2021_01_11_13_53_08",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2020_12_22_13_36_25",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2020_12_19_14_36_38",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2020_12_18_08_49_21",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2020_12_17_16_02_30",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2020_09_28_20_53_15",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2020_10_15_12_21_33",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2020_10_15_12_21_33",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2020_11_05_10_36_20",

                # These are mostly lower resolution world and eye videos (Not processed yet)
                # "/hdd01/kamran_sync/vedbcloud0/staging/2020_08_23_19_43_29",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2020_08_23_20_05_26",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2020_08_23_22_27_12",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2020_09_14_13_54_11",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2020_09_19_14_43_04",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2020_09_22_16_08_45",

                # "/hdd01/kamran_sync/vedbcloud0/staging/2021_02_04_16_42_40",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2021_02_04_17_01_18",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2020_10_19_00_23_14",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2020_10_15_12_05_24",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2020_10_15_12_30_26",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2020_10_16_11_07_40",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2020_11_03_02_27_22",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2020_12_08_06_39_12",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2020_12_12_03_14_20",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2020_12_13_11_12_10",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2020_12_18_12_04_33",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2021_01_11_16_33_39",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2021_01_18_16_56_46",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2021_01_20_11_32_15",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2021_02_01_12_35_10",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2021_02_02_11_14_14",
                # "/hdd01/kamran_sync/vedbcloud0/staging/2020_11_03_04_50_44",
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
                    session_folder, param_dict, batch_size_pupil=1000, batch_size_marker=100)
        except:
           print("Failed for session %s " % session_folder)
        # else:
        #     print("skipped session: ", session_folder)

        print("Gaze Calibratrion Result: ", result)
