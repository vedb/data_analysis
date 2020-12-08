import os
import sys
import glob
from data_analysis import gaze

# Directory for the recording sessions
fdir = "/hdd01/kamran_sync/vedbcloud0/"
next_in_line = ["2020_08_23_19_43_29", "000", "2020_08_23_20_05_26", "2020_08_23_22_27_12", "2020_09_14_13_54_11", "2020_10_15_12_30_26"]
next_in_line = ["2020_08_23_19_43_29"]
done = ["2020_10_15_12_05_24", "2020_10_19_00_23_14", "2020_10_16_11_07_40", "2020_10_16_11_26_00"]
if __name__ == "__main__":
    sessions = glob.glob(fdir + "*")
    print("all sessions: ", sessions)
    for session_folder in sessions:
        for directory in next_in_line:
            if directory in session_folder:
                print("running analysis for:", session_folder)
                #try:
                out = gaze.pipelines.pupil_2d_binocular_v01(
                    session_folder, batch_size_pupil=200, batch_size_marker=50
                )
                break
                #except:
                #    print("Failed for session %s" % session_folder)
            else:
                print("skipping analysis for:", session_folder)
