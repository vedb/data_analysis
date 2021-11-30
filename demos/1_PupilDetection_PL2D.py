from data_analysis import gaze
from data_analysis.session import Session
from pathlib import Path
from data_analysis.gaze.pupil_detection_pl import read_process_configs
from data_analysis.gaze.pupil_detection_pl import create_output_data
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
        my_session.gaze_saving_path = gaze_directory + session_id
        Path(my_session.gaze_saving_path).mkdir(parents=True, exist_ok=True)
        N = viz_params['visualization']['number_of_eye_frames']
        eye0_video = my_session.open_video_cv2("eye0")
        eye1_video = my_session.open_video_cv2("eye1")
        # Todo: figure out whether these few extra frames are necessary or not
        total_frame_number = max(int(my_session.video_total_frame_count_cv2(eye0_video)) + 10,
                                 int(my_session.video_total_frame_count_cv2(eye1_video)) + 10)
        my_session.close_video_cv2("eye0")
        my_session.close_video_cv2("eye1")

        df_index = np.arange(0, 2*total_frame_number)
        df = create_output_data(df_index)
        process = ProcessPupilDetection(target=run_pupil_detection_PL,
                                        name="Pupil Tracking",
                                        arguments=(my_session, df))
        process.start()
        # Todo: Pass a meaningful timeout to the process wait
        process.join(120*60)
        if process.is_alive():
            print("Still alive, going to terminate!!")
            process.terminate()
        else:
            print("{} Pupil Detection Done!".format("PL"))
            print("Process output File:\n {}".format(pd.read_pickle(my_session.gaze_saving_path +
                                                                    "/pupil_positions_PL.pkl").head(10)))
