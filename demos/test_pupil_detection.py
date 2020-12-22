import os
import sys
import glob
from data_analysis import gaze
from os import listdir
from os.path import isfile, join

if __name__ == "__main__":
    # Directory for the recording sessions
    fdir = "/hdd01/Deep_Gaze_Tracking/"
    #next_in_line = ["2020_08_23_19_43_29"]
    #done = ["2020_10_15_12_05_24", "2020_10_19_00_23_14", "2020_10_16_11_07_40", "2020_10_16_11_26_00"]
    onlyfiles = [f for f in listdir(fdir) if isfile(join(fdir, f))]
    print("all files: ", onlyfiles)
    for file_name in onlyfiles:
        print("running analysis for:", file_name)
        #try:
        out = gaze.pipelines.pupil_2d_monocular_v02(file_name,
            fdir, batch_size_pupil=200)
        print("Returned: ", out)

    print("Done!")
