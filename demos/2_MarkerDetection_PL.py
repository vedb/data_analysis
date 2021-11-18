from data_analysis import gaze
from data_analysis.session import Session
from pathlib import Path

from data_analysis.scene.marker_detection_pl import read_process_configs
from data_analysis.scene.marker_detection_pl import create_output_data
from data_analysis.scene.marker_detection_pl import ProcessMarkerDetection
from data_analysis.scene.marker_detection_pl import run_marker_detection_ved
import numpy as np
import pandas as pd


pipeline_param_path = "../config/gaze_calibration_parameters.yaml"
sessions_param_path = "../config/journal_sessions_list.yaml"
viz_param_path = "../config/visualization_parameters.yaml"


if __name__ == "__main__":

    pipeline_params, session_params, viz_params = read_process_configs(pipeline_param_path,
                                                                       sessions_param_path,
                                                                       viz_param_path)
    session_directory = pipeline_params["directory"]["session_directory"]
    gaze_directory = pipeline_params["directory"]["gaze_directory"]
    sessions = session_params["sessions"]
    world_scale = viz_params["visualization"]["world_scale"]

    print("total number of sessions", len(sessions))
    for session_id in sessions:
        print("running analysis for:", session_id)
        my_session = Session(session_directory + session_id)
        my_session.gaze_saving_path = gaze_directory + session_id
        Path(my_session.gaze_saving_path).mkdir(parents=True, exist_ok=True)

        calibration_start = session_params["sessions"][session_id]["calibration_0_start"]
        start_index = int(calibration_start[0][0])*1800 + int(calibration_start[0][1])*30

        calibration_end = session_params["sessions"][session_id]["calibration_0_end"]
        end_index = int(calibration_end[0][0])*1800 + int(calibration_end[0][1])*30

        df_index = np.arange(0, end_index - start_index)
        df = create_output_data(df_index)
        marker_type = "calibration"
        process = ProcessMarkerDetection(target=run_marker_detection_ved,
                                         name="Calibration Marker Tracking",
                                         arguments=(my_session, df, marker_type, world_scale, start_index, end_index))
        process.start()
        # Todo: Pass a meaningful timeout to the process wait
        process.join(20*60)
        if process.is_alive():
            print("Still alive, going to terminate!!")
            process.terminate()
        else:
            print("{} Marker Detection Done!".format(marker_type))
            print("Process output File:\n {}".format(pd.read_pickle(my_session.gaze_saving_path +
                                                                    "/world_calibration_marker.pkl").head(10)))

        validation_start = session_params["sessions"][session_id]["validation_0_start"]
        start_index = int(validation_start[0][0])*1800 + int(validation_start[0][1])*30

        validation_end = session_params["sessions"][session_id]["validation_0_end"]
        end_index = int(validation_end[0][0])*1800 + int(validation_end[0][1])*30

        df_index = np.arange(0, end_index - start_index)
        df = create_output_data(df_index, marker_type="validation", size=(6, 8))
        marker_type = "validation"
        process = ProcessMarkerDetection(target=run_marker_detection_ved,
                                         name="Validation Marker Tracking",
                                         arguments=(my_session, df, marker_type, world_scale, start_index, end_index))
        process.start()
        # Todo: Pass a meaningful timeout to the process wait
        process.join(20*60)
        if process.is_alive():
            print("Still alive, going to terminate!!")
            process.terminate()
        else:
            print("{} Marker Detection Done!".format(marker_type))
            print("Process output File:\n {}".format(pd.read_pickle(my_session.gaze_saving_path +
                                                                    "/world_validation_marker.pkl").head(10)))
