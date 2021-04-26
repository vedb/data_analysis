import os
import sys
import glob

import numpy as np
import yaml
from data_analysis import gaze
from data_analysis import visualization
from data_analysis.visualization.gaze_quality import plot_gaze_accuracy_heatmap
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
    cumulative_file = param_dict['directory']['cumulative_gaze_file']
    all_subject_data = np.load(cumulative_file, allow_pickle=True)

    number_of_sessions = all_subject_data['calibration_points'].shape[0]
    print("total number of sessions", number_of_sessions)
    for i in range(number_of_sessions):
        print("==> shapes: ", all_subject_data['calibration_points'][i].shape,
              all_subject_data['calibration_gaze'][i].shape,
              all_subject_data['validation_points'][i].shape,
              all_subject_data['validation_gaze'][i].shape )
        if i == 0:
            calibration_points = all_subject_data['calibration_points'][i]
            calibration_gaze = all_subject_data['calibration_gaze'][i]
            validation_points = all_subject_data['validation_points'][i]
            validation_gaze = all_subject_data['validation_gaze'][i]
        else:
            calibration_points = np.vstack((calibration_points, all_subject_data['calibration_points'][i]))
            calibration_gaze = np.vstack((calibration_gaze, all_subject_data['calibration_gaze'][i]))
            if len(all_subject_data['validation_points'][i]) > 0:
                validation_points = np.vstack((validation_points, all_subject_data['validation_points'][i]))
            if len(all_subject_data['validation_gaze'][i]) > 0:
                validation_gaze = np.vstack((validation_gaze, all_subject_data['validation_gaze'][i]))
        print("Shapes: ", i, calibration_points.shape, calibration_gaze.shape, validation_points.shape, validation_gaze.shape)

    save_directory = "/hdd01/kamran_sync/lab/students/kamran/Projects/Results/"
    marker_type = "Calibration"
    plot_gaze_accuracy_heatmap(calibration_points, calibration_gaze,None, file_name=save_directory+"calibration_accuracy.png",reference_type=marker_type)
    marker_type = "Validation"
    plot_gaze_accuracy_heatmap(validation_points, validation_gaze,None, file_name=save_directory+"validation_accuracy.png",reference_type=marker_type)
