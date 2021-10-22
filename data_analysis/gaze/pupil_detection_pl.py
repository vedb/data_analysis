import pupil_detectors
import numpy as np
from pupil_detectors import Detector2D, Detector3D
from data_analysis.process import BaseProcess
import cv2
import pandas as pd
import random
from tqdm import tqdm

class PupilDetection_PL(BaseProcess):

    def __init__(self, name, session, *args, **kwargs):
        """ Constructor.

        Parameters
        ----------
        session: VED Session Object
            This include pointers to all streams of data
            and load/save mechanisms

        name: str
            The name of the process

        data_structure: str
            Data Structure format i.e. "pandas" or "array"

        data_size: int
            Number of rows of the output data structure
            i.e. pd.DataFrame()

        data_size: int
            Number of rows of the output data structure
            i.e. pd.DataFrame()

        args:
            Additional arguments

        kwargs:
            Additional keyword arguments
        """
        self.session = session
        super().__init__(name, data_structure="pandas", data_size=100, *args, **kwargs)
        self.output_data = self.create_output_data()
        self.args = args

    def create_output_data(self):
        df_keys = ["pupil_timestamp", "eye_index", "world_index", "eye_id", "confidence", "norm_pos_x",
                           "norm_pos_y",
                           "diameter", "method", "ellipse_center_x", "ellipse_center_y", "ellipse_axis_a", "ellipse_axis_b",
                           "ellipse_angle", "diameter_3d", "model_confidence", "model_id", "sphere_center_x",
                           "sphere_center_y", "sphere_center_z", "sphere_radius", "circle_3d_center_x",
                           "circle_3d_center_y", "circle_3d_center_z", "circle_3d_normal_x", "circle_3d_normal_y",
                           "circle_3d_normal_z", "circle_3d_radius", "theta", "phi", "projected_sphere_center_x",
                           "projected_sphere_center_y", "projected_sphere_axis_a", "projected_sphere_axis_b",
                           "projected_sphere_angle"]
        # Because we want to run pupil detection on both right and left eye videos we need 2N rows
        df = pd.DataFrame(columns=df_keys, index=np.arange(2*self.data_size), dtype=np.float64)
        return df

    def target_method(self):
        def f(df, session):
            if session.eye0_video_isOpen:
                eye0_video = session.eye0_video
            else:
                eye0_video = session.open_video_cv2(video_name="eye0")
            if session.eye1_video_isOpen:
                eye1_video = session.eye1_video
            else:
                eye1_video = session.open_video_cv2(video_name="eye1")

            eye0_frame_size = session.video_frame_size_cv2(eye0_video)
            eye1_frame_size = session.video_frame_size_cv2(eye1_video)

            eye0_timestamp = session.read_timestamp_np("eye0")
            eye1_timestamp = session.read_timestamp_np("eye1")

            properties = {}
            detector_2D = Detector2D(properties=properties)
            detection_method = "2d c++"

            eye_image_indexes = df.index.values
            print("Running pupil tracking for {} eye video frames".format(len(eye_image_indexes)))
            count = eye_image_indexes[0]
            eye0_video.set(1, count)
            eye1_video.set(1, count)
            for frame_index in tqdm(eye_image_indexes):
                ret_0, img_0 = eye0_video.read()
                ret_1, img_1 = eye1_video.read()
                if not ret_0 or not ret_1:
                    break
                if np.ndim(img_0) == 3:
                    img_0 = img_0[:, :, 0]
                if np.ndim(img_1) == 3:
                    img_1 = img_1[:, :, 0]
                if count == 0:
                    prev_frame_0 = np.array(img_0, dtype=np.float64)
                    prev_frame_1 = np.array(img_1, dtype=np.float64)
                else:
                    prev_frame_0 = (img_0 + prev_frame_0) / 2.0
                    prev_frame_1 = (img_1 + prev_frame_1) / 2.0

                pupil0_dict_2d = detector_2D.detect(np.ascontiguousarray(img_0))
                df["pupil_timestamp"].loc[count] = eye0_timestamp[frame_index]
                df["eye_index"].loc[count] = int(frame_index)
                df["eye_id"].loc[count] = 0
                df["confidence"].loc[count] = pupil0_dict_2d["confidence"]
                df["norm_pos_x"].loc[count] = pupil0_dict_2d["location"][0] / eye0_frame_size[0]
                df["norm_pos_y"].loc[count] = pupil0_dict_2d["location"][1] / eye0_frame_size[1]
                df["diameter"].loc[count] = pupil0_dict_2d["diameter"]
                df["method"].loc[count] = detection_method
                df["ellipse_center_x"].loc[count] = pupil0_dict_2d["ellipse"]["center"][0]
                df["ellipse_center_y"].loc[count] = pupil0_dict_2d["ellipse"]["center"][1]
                df["ellipse_axis_a"].loc[count] = pupil0_dict_2d["ellipse"]["axes"][0]
                df["ellipse_axis_b"].loc[count] = pupil0_dict_2d["ellipse"]["axes"][1]
                df["ellipse_angle"].loc[count] = pupil0_dict_2d["ellipse"]["angle"]
                count = count + 1

                pupil1_dict_2d = detector_2D.detect(np.ascontiguousarray(img_1))
                df["pupil_timestamp"].loc[count] = eye1_timestamp[frame_index]
                df["eye_index"].loc[count] = int(frame_index)
                df["eye_id"].loc[count] = 1
                df["confidence"].loc[count] = pupil1_dict_2d["confidence"]
                df["norm_pos_x"].loc[count] = pupil1_dict_2d["location"][0] / eye0_frame_size[0]
                df["norm_pos_y"].loc[count] = pupil1_dict_2d["location"][1] / eye0_frame_size[1]
                df["diameter"].loc[count] = pupil1_dict_2d["diameter"]
                df["method"].loc[count] = detection_method
                df["ellipse_center_x"].loc[count] = pupil1_dict_2d["ellipse"]["center"][0]
                df["ellipse_center_y"].loc[count] = pupil1_dict_2d["ellipse"]["center"][1]
                df["ellipse_axis_a"].loc[count] = pupil1_dict_2d["ellipse"]["axes"][0]
                df["ellipse_axis_b"].loc[count] = pupil1_dict_2d["ellipse"]["axes"][1]
                df["ellipse_angle"].loc[count] = pupil1_dict_2d["ellipse"]["angle"]
                count = count + 1
            df.to_pickle("~/Desktop/pupil_data.pkl")
            df.to_csv("~/Desktop/pupil_data.csv")

            session.close_video_cv2("eye0")
            session.close_video_cv2("eye1")
            cv2.destroyAllWindows()
            print("\nDone!")
            return df.values.tolist()
        print("Running Process {} on {} for {} frames".format(self.name, self.target_method, self.data_size))
        # self.result = self.pool.apply_async(self.target_method, args=self.args)
        self.result = self.pool.map(f, self.args)
        return self.result




# def plabs_detect_pupil(
#     eye_video, timestamps=None, progress_bar=None, id=None, properties=None, **kwargs
# ):
#     """
#
#     This is a simple wrapper to allow Pupil Labs `pupil_detectors` code
#     to process a whole video of eye data.
#
#     Parameters
#     ----------
#     eye_video : array
#         video to parse,  (t, x, y, [color]), already loaded as uint8 data.
#     timestamps : array
#         optional, time associated with each frame in video. If provided,
#         timestamps for each detected pupil are returned with
#     progress_bar : tqdm object or None
#         if tqdm object is provided, a progress bar is displayed
#         as the video is processed.
#     id : int, 0 or 1
#         ID for eye (eye0, i.e. left, or eye1, i.e. right)
#
#     Notes
#     -----
#     Parameters for Pupil Detector2D object, passed as a dict called
#     `properties` to `pupil_detectors.Detector2D()`; fields are:
#         coarse_detection = True
#         coarse_filter_min = 128
#         coarse_filter_max = 280
#         intensity_range = 23
#         blur_size = 5
#         canny_treshold = 160
#         canny_ration = 2
#         canny_aperture = 5
#         pupil_size_max = 100
#         pupil_size_min = 10
#         strong_perimeter_ratio_range_min = 0.6
#         strong_perimeter_ratio_range_max = 1.1
#         strong_area_ratio_range_min = 0.8
#         strong_area_ratio_range_max = 1.1
#         contour_size_min = 5
#         ellipse_roundness_ratio = 0.09    # HM! Try setting this?
#         initial_ellipse_fit_treshhold = 4.3
#         final_perimeter_ratio_range_min = 0.5
#         final_perimeter_ratio_range_max = 1.0
#         ellipse_true_support_min_dist = 3.0
#         support_pixel_ratio_exponent = 2.0
#     Returns
#     -------
#     pupil_dicts : list of dicts
#         dictionary for each detected instance of a pupil. Each
#         entry has fields:
#         luminance
#         timestamp [if timestamps are provided, which they should be]
#         norm_pos
#     """
#     if progress_bar is None:
#         progress_bar = lambda x: x
#     # Specify detection method later?
#     if properties is None:
#         properties = {}
#     det = pupil_detectors.Detector2D(properties=properties)
#     n_frames, eye_vdim, eye_hdim = eye_video.shape[:3]
#     eye_dims = np.array([eye_hdim, eye_vdim])
#     pupil_dicts = []
#     lum = np.zeros((n_frames,))
#     for frame in progress_bar(range(n_frames)):
#         fr = eye_video[frame].copy()
#         # If an extra dimension is present, assume eye video is spuriously RGB
#         if np.ndim(fr) == 3:
#             fr = fr[:, :, 0]
#         # Pupil needs c-ordered arrays, so switch from default load:
#         fr = np.ascontiguousarray(fr)
#         # Call detector & process output
#         out = det.detect(fr)
#         # Get rid of raw data as input; no need to keep
#         if "internal_2d_raw_data" in out:
#             _ = out.pop("internal_2d_raw_data")
#         # Save average luminance of eye video for reference
#         out["luminance"] = fr.mean()
#         # Normalized position
#         out["norm_pos"] = (np.array(out["location"]) / eye_dims).tolist()
#         if timestamps is not None:
#             out["timestamp"] = timestamps[frame]
#         if id is not None:
#             out["id"] = id
#         pupil_dicts.append(out)
#     return pupil_dicts
