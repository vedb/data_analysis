from data_analysis import gaze
from data_analysis.session import Session

from data_analysis.gaze.pupil_detection_pl import read_process_configs
from data_analysis.gaze.pupil_detection_pl import creat_output_data
from data_analysis.gaze.pupil_detection_pl import ProcessPupilDetection
from data_analysis.gaze.pupil_detection_pl import run_pupil_detection_PL
import numpy as np
import pandas as pd


pipeline_param_path = "../config/gaze_calibration_parameters.yaml"
sessions_param_path = "../config/journal_sessions_list.yaml"
viz_param_path = "../config/visualization_parameters.yaml"


if __name__ == "__main__":

    pipeline_params, session_params, viz_params = read_process_configs(pipeline_param_path,
                                                                       sessions_param_path,
                                                                       viz_param_path)
    session_directory = pipeline_params['directory']['session_directory']
    gaze_directory = pipeline_params['directory']['gaze_directory']
    sessions = session_params['sessions']

    print("total number of sessions", len(sessions))
    for session_id in sessions:
        print("running analysis for:", session_id)
        my_session = Session(session_directory + session_id)
        N = viz_params['visualization']['number_of_eye_frames']

        df_index = np.arange(0, N)
        df = creat_output_data(df_index)
        process = ProcessPupilDetection(target=run_pupil_detection_PL,
                                        name="Pupil Tracking",
                                        arguments=(my_session, df))
        process.start()
        # Todo: Pass a meaningful timeout to the process wait
        process.join(5*60)
        if process.is_alive():
            print("Still alive, going to terminate!!")
            process.terminate()
        print("Pupil Detection Done!")
        print("Process output:\n {}".format(pd.read_pickle("~/Desktop/pupil_data.pkl").head()))
