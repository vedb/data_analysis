from data_analysis import gaze
from data_analysis.session import Session
from data_analysis.utils import read_yaml
from data_analysis.process import BaseProcess
from data_analysis.gaze.pupil_detection_pl import run_pupil_detection_PL
# from data_analysis.gaze.pupil_detection_pl import creat_output_data
# from data_analysis.gaze.pupil_detection_pl import collect_results
import multiprocessing as mp
import numpy as np
import pandas as pd
from data_analysis.gaze.pupil_detection_pl import read_process_configs
import os
from data_analysis.gaze.calibrate_pl import (read_eye_calibration_data,
                                             get_right_pupil_DF,
                                             get_left_pupil_DF,
                                             get_ref_DF,
                                             run_calibration_PL,
                                             run_gaze_mapper_PL,
                                             clean_up_gaze_df)


pipeline_param_path = "../config/gaze_calibration_parameters.yaml"
sessions_param_path = "../config/journal_sessions_list.yaml"
viz_param_path = "../config/visualization_parameters.yaml"


if __name__ == "__main__":
    # pipeline_params = read_yaml(pipeline_param_path)
    # session_directory = pipeline_params['directory']['session_directory']
    # gaze_directory = pipeline_params['directory']['gaze_directory']
    #
    # session_params = read_yaml(sessions_param_path)
    # sessions = session_params['sessions']
    #
    # viz_params = read_yaml(viz_param_path)
    # results = []
    # print("total number of sessions", len(sessions))
    # for session_id in sessions:
    #     print("running analysis for:", session_id)
    #     my_session = Session(session_directory + session_id)
    #     N = viz_params['visualization']['number_of_eye_frames']
    #
    #     # eye0_video = my_session.open_video_cv2("eye0")
    #     # eye1_video = my_session.open_video_cv2("eye1")
    #     # max_frame_number = min(int(my_session.video_total_frame_count_cv2(eye0_video)) - 10,
    #     #                        int(my_session.video_total_frame_count_cv2(eye0_video)) - 10)
    #
    #     df_index = np.arange(0, N)
    #     df = creat_output_data(df_index)
    #     # pool = mp.Pool(processes=mp.cpu_count())
    #     process = mp.Process(target=run_pupil_detection_PL, name="Pupil Tracking", args=(my_session, df))
    #     # pool.apply_async(run_pupil_detection_PL, args=(my_session, output_df, N,), callback=collect_results)
    #     # pool.close()
    #     # pool.join()
    #     process.start()
    #     process.join()
    #     print(pd.read_pickle("/home/kamran/Desktop/pupil_data.pkl").head())
    #     # pupil_dataframe, mean_image_0, mean_image_1 = run_pupil_detection_PL(session=my_session, number_of_frames=N)
    #     #
    #     # result =
    #     # result = gaze.pipelines.pupil_2d_binocular_v01(
    #     #         session_folder, param_dict, batch_size_pupil=500, batch_size_marker=100)
    #
    #     print("Pupil Detection Done!: ")

    pipeline_param_path = "/home/kamran/Code/data_analysis/config/gaze_calibration_parameters.yaml"
    sessions_param_path = "/home/kamran/Code/data_analysis/config/journal_sessions_list.yaml"
    viz_param_path = "/home/kamran/Code/data_analysis/config/visualization_parameters.yaml"

    # gaze_path = "/hdd01/kamran_sync/processed/gaze_pilot/"
    # viz_path = "/hdd01/kamran_sync/processed/visualization//"

    pipeline_params, session_params, viz_params = read_process_configs(pipeline_param_path,
                                                                       sessions_param_path,
                                                                       viz_param_path)
    session_directory = pipeline_params['directory']['session_directory']
    gaze_directory = pipeline_params['directory']['gaze_directory']
    sessions = session_params['sessions']

    print("total number of sessions", len(sessions))
    for session_id in sessions:
        print("Running for Session {}".format(session_id))
        processed_path = os.path.join(gaze_path, session_id)
        p_df, cal_df, val_df = read_eye_calibration_data(processed_path)
        print("Frame Counts: {} Pupils, {} Calibrations, {} Validations".format(len(p_df), len(cal_df), len(val_df)))

        right_pupil_df = get_right_pupil_DF(p_df)
        left_pupil_df = get_left_pupil_DF(p_df)
        ref_df = get_ref_DF(cal_df)

        pupil_list_left = left_pupil_df.to_dict('records')
        pupil_list_right = right_pupil_df.to_dict('records')
        ref_list = ref_df.to_dict('records')

        method, result, pupil_list_binocular = run_calibration_PL(pupil_list_right,
                                                                  pupil_list_left,
                                                                  ref_list)

        final_result, gaze_binocular = run_gaze_mapper_PL(result, pupil_list_binocular)
        if final_result:
            gaze_df = clean_up_gaze_df(gaze_binocular)
            gaze_df.to_pickle(os.path.join(processed_path, "gaze_positions_PL.pkl"))
            gaze_df.to_csv(os.path.join(processed_path, "gaze_positions_PL.csv"))
        else:
            gaze_df = None
            print("\n\nGaze Mapping Failed for {}".format(session_id))
            print("What should be done when the mapping fails!!?")