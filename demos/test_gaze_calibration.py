import os
import sys
import glob
from data_analysis import gaze

# Directory for the recording sessions
fdir = "/home/veddy06/recordings/2020_10_15_12_05_24/"

if __name__ == "__main__":
    sessions = glob.glob(fdir + "*")
    print("all sessions: ", sessions)
    for session_folder in sessions:
        print("running analysis for:", session_folder)
        try:
            out = gaze.pipelines.pupil_monocular_v01(
                session_folder, batch_size_pupil=100, batch_size_marker=20
            )
        except:
            print("Failed for session %s" % session_folder)
