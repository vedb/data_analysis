# import vm_preproc as vmp
# import vedb_store
import numpy as np
import tqdm
import os
# from ..gaze import gaze_utils
# from ..gaze import marker_detection
import cv2
import imageio
import itertools

from .checkerboard_detection import detect_checkerboard
def clean_up_reference_dict(val_reference_arrays):
    a = []
    t = []
    for data in val_reference_arrays['location']:
        if len(data)>0:
            # print("size change", data.shape, data.flatten().shape, len(a))
            a.append(data.flatten())
    a = list(itertools.chain(*a))
    a = np.array(a)
    print(a.shape)
    frame_count = int(len(a)/(48*2))
    b = np.reshape(a,(frame_count, 48,2))
    print(b.shape)

    for data in val_reference_arrays['timestamp']:
        if len(data) > 0:
            t.append(data.flatten())
    t = list(itertools.chain(*t))
    t = np.array(t)
    print('time_stamp size:',t.shape)
    return b, t


def create_slippage_index_array(sessions_dict, session_id):
    this_session = sessions_dict['sessions'][session_id]
    frame_index = np.array([])
    start_index = this_session['validation_start'][0][0] * 60 + this_session['validation_start'][0][1]
    end_index = this_session['validation_end'][0][0] * 60 + this_session['validation_end'][0][1]
    frame_index = np.append(frame_index, np.arange(start_index * 30, end_index * 30))

    start_index = this_session['slow_validation_start'][0][0] * 60 + this_session['slow_validation_start'][0][1]
    end_index = this_session['slow_validation_end'][0][0] * 60 + this_session['slow_validation_end'][0][1]
    frame_index = np.append(frame_index, np.arange(start_index * 30, end_index * 30))

    start_index = this_session['medium_validation_start'][0][0] * 60 + this_session['medium_validation_start'][0][1]
    end_index = this_session['medium_validation_end'][0][0] * 60 + this_session['medium_validation_end'][0][1]
    frame_index = np.append(frame_index, np.arange(start_index * 30, end_index * 30))

    start_index = this_session['fast_validation_start'][0][0] * 60 + this_session['fast_validation_start'][0][1]
    end_index = this_session['fast_validation_end'][0][0] * 60 + this_session['fast_validation_end'][0][1]
    frame_index = np.append(frame_index, np.arange(start_index * 30, end_index * 30))

    start_index = this_session['slow_start'][0][0] * 60 + this_session['slow_start'][0][1]
    end_index = this_session['slow_end'][0][0] * 60 + this_session['slow_end'][0][1]
    frame_index = np.append(frame_index, np.arange(start_index * 30, end_index * 30))

    start_index = this_session['medium_start'][0][0] * 60 + this_session['medium_start'][0][1]
    end_index = this_session['medium_end'][0][0] * 60 + this_session['medium_end'][0][1]
    frame_index = np.append(frame_index, np.arange(start_index * 30, end_index * 30))

    start_index = this_session['fast_start'][0][0] * 60 + this_session['fast_start'][0][1]
    end_index = this_session['fast_end'][0][0] * 60 + this_session['fast_end'][0][1]
    frame_index = np.append(frame_index, np.arange(start_index * 30, end_index * 30))

    return frame_index.flatten()


def detect_markers(
    session_directory,
    session_folder,
    param_dict,
    sessions_dict,
    world_scale=1,
    progress_bar=tqdm.tqdm,
):
    """
    Parameters
    ----------
    tag : A short label for the pipeline
    session_folder : string
        file path to session. Ultimately want to replace this with a session
        object from the database.
    string_name : format string
        must contain {step}; if provided, tag and output_path are ignored

    Notes
    -----
    Ultimately, for saving files, we want the output of each step saved
    along with the function and parameters that were used to generate it.

    This is not the case now. Needs work.
    """
    print(param_dict.keys())
    # Todo: Read length of the session id from parameters?
    session_id = session_folder[-19:]
    output_path = param_dict['directory']['saving_directory'] + session_id + '/'
    processed_path = param_dict['directory']['gaze_directory'] + session_id + '/'
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    # Deal with inputs
    if output_path is None:
        #output_path = session_folder
        raise ValueError("parameters' yaml file doesn't have valid saving_directory!")
    else:
        print("saving results to: ", output_path)


    start_time = param_dict['visualization']['start_time']
    end_time = param_dict['visualization']['end_time']
    run_for_all_frames = param_dict['visualization']['run_for_all_frames']
    save_video = param_dict['visualization']['save_world_output']
    world_scale = param_dict['visualization']['world_scale']

    gaze_calibration_tag = param_dict['calibration']['pupil_detection'] + '_' +\
          param_dict['calibration']['eye'] + '_' +\
          param_dict['calibration']['algorithm']

    tag = 'world_' +\
          str(start_time[0])+ '_' +\
          str(start_time[1]) + '_' +\
          str(end_time[0])+ '_' +\
          str(end_time[1])

    # Todo: Read this from session info
    fps = 30
    print('tag : ', tag)
    string_name = os.path.join(output_path, tag + "_{step}.npz")
    print("file_name", string_name)
    if not os.path.exists(output_path):
        print("creating", output_path)
        os.makedirs(output_path)
    # (0) Get session
    # session = vedb_store.Session(folder=session_folder)
    # (1) Read Start and End Indexes
    start_index = (start_time[0] * 60 + start_time[1]) * fps
    end_index = (end_time[0] * 60 + end_time[1]) * fps

    # (2) Read Gaze File
    # gaze_file = processed_path + '/' + gaze_calibration_tag + '_gaze.npz'
    #
    # if os.path.exists(gaze_file):
    #     print("Loading gaze data")
    #     gaze_arrays = {}
    #     dat = np.load(gaze_file, allow_pickle=True)
    #     for k in dat.keys():
    #         gaze_arrays[k] = dat[k]
    #     gaze_list = gaze_utils.arraydict_to_dictlist(gaze_arrays)
    # else:
    #     raise ValueError("No valid gaze file was found!", gaze_file)
    # for key, value in gaze_list[0].items():
    #     print(key, value)

    # (3) Read Video File
    world_video_file = param_dict['directory']['session_directory'] + session_id + "/world.mp4"
    print("world Video File: ", world_video_file)
    vid = imageio.get_reader(world_video_file, "ffmpeg")
    cap = cv2.VideoCapture(world_video_file)
    world_time_stamp_file = param_dict['directory']['session_directory'] + session_id + "/world_timestamps.npy"
    world_time_stamp = np.load(world_time_stamp_file)
    if run_for_all_frames:
        total_frame_numbers = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_index = 0
        end_index = total_frame_numbers
        print("Total Number of Frames: ", total_frame_numbers)
    this_session = sessions_dict['sessions'][session_id]
    frame_index_array = create_slippage_index_array(sessions_dict, session_id)
    imgpoints, marker_found = detect_checkerboard(path=session_folder, checkerboard_size=(6,8), scale=world_scale, frame_index_array=frame_index_array, this_session=this_session)
    print("marker point size: ", np.array(imgpoints).shape)
    print("marker flag size: ", np.array(marker_found).shape)
    marker_file_name = processed_path + "validation_points.npy"
    np.save(marker_file_name, imgpoints)
    flag_file_name = processed_path + "validation_found.npy"
    np.save(flag_file_name, marker_found)

    # gaze_timestamp = []
    # for my_index in range(len(gaze_list)):
    #     gaze_timestamp.append(gaze_list[my_index]['gaze_binocular']['timestamp'])


    # (4) Read Calibration Marker File
    # reference_file = processed_path + '/' + gaze_calibration_tag + '_calibration_ref_pos.npz'
    # if os.path.exists(reference_file):
    #     print("Loading reference data: Calibration")
    #     cal_reference_arrays = {}
    #     dat = np.load(reference_file, allow_pickle=True)
    #     for k in dat.keys():
    #         cal_reference_arrays[k] = dat[k]
    #     reference_list = gaze_utils.arraydict_to_dictlist(cal_reference_arrays)
    # else:
    #     raise ValueError("No valid reference/calibration file was found!", reference_file)
    # for key, value in reference_list[0].items():
    #     print(key, value)
    # cal_reference_timestamps = cal_reference_arrays['timestamp']

    # (5) Read Validation Marker File
    # reference_file = processed_path + '/' + gaze_calibration_tag + '_validation_ref_pos.npz'
    # if os.path.exists(reference_file):
    #     print("Loading reference data: Validation")
    #     val_reference_arrays = {}
    #     dat = np.load(reference_file, allow_pickle=True)
    #     for k in dat.keys():
    #         val_reference_arrays[k] = dat[k]
    #     reference_list = gaze_utils.arraydict_to_dictlist(val_reference_arrays)
    # else:
    #     raise ValueError("No valid reference/validation file was found!", reference_file)
    # for key, value in reference_list[0].items():
    #     print(key, value)
    # val_pos, val_reference_timestamps = clean_up_reference_dict(val_reference_arrays)
    #
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * world_scale)
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * world_scale)
    # video_size = (frame_width, frame_height)
    # print("frame size:", video_size)
    # print("First Frame = %d" % start_index)
    # print("Last Frame = %d" % end_index)
    # print("scale[x,y] = ", world_scale)

    # Instantiate the video recorder in order to store the processed images to an output video
    # if save_video:
    #     fourcc = 'mp4v'
    #     output_video_file = os.path.join(output_path, str(tag)) +".mp4"
    #     print("output video file: ", output_video_file)
    #     out_video = cv2.VideoWriter(output_video_file,cv2.VideoWriter_fourcc(*fourcc), fps, video_size)

    # cal_points = []
    # cal_gaze = []
    # val_points = []
    # val_gaze = []
    # # Read the next frame from the video.
    # for i in range(start_index, end_index):
    #
    #     print("frame index= ", i ,end='\r')
    #     gaze_index = np.argmin(np.abs((gaze_timestamp - world_time_stamp[i]).astype(float)))
    #     gaze_norm_x = gaze_list[gaze_index]['gaze_binocular']['norm_pos'][0]
    #     gaze_norm_y = gaze_list[gaze_index]['gaze_binocular']['norm_pos'][1]
    #
    #     gaze_pixel_x = int(gaze_norm_x * frame_width)
    #     gaze_pixel_y = int((gaze_norm_y) * frame_height)
    #     #frame_no_gaze = img.copy()
    #     if len(val_reference_timestamps)>0:
    #         if (min(np.abs(val_reference_timestamps - world_time_stamp[i]))<0.03):
    #             reference_index = np.argmin(np.abs(val_reference_timestamps - world_time_stamp[i]))
    #             # ref_pixel_x = int(cal_reference_arrays['location'][reference_index, 0] * world_scale)
    #             # ref_pixel_y = int(cal_reference_arrays['location'][reference_index, 1] * world_scale)
    #             val_pos[reference_index, :, 0] = val_pos[reference_index, :, 0] * frame_width * (2)
    #             val_pos[reference_index, :, 1] = val_pos[reference_index, :, 1] * frame_height * (1)
    #             ref_corners = val_pos[reference_index, :, :]
    #             p = val_pos[reference_index, :]
    #             p[:,0] = p[:,0]*2
    #             val_points.append(np.mean(p,axis=0))
    #             val_gaze.append([gaze_norm_x,gaze_norm_y])
    #
    #
    #     if (min(np.abs(cal_reference_timestamps - world_time_stamp[i]))<0.03):
    #         reference_index = np.argmin(np.abs(cal_reference_timestamps - world_time_stamp[i]))
    #         ref_pixel_x = int(cal_reference_arrays['location'][reference_index, 0] * world_scale)
    #         ref_pixel_y = int(cal_reference_arrays['location'][reference_index, 1] * world_scale)
    #         cal_points.append([cal_reference_arrays['location'][reference_index, 0], cal_reference_arrays['location'][reference_index, 1]])
    #         cal_gaze.append([gaze_norm_x,gaze_norm_y])
    #
    # val_points = np.array(val_points)
    # val_gaze = np.array(val_gaze)
    # cal_points = np.array(cal_points)
    # cal_gaze = np.array(cal_gaze)
    # print("\nDone!")
    # print("cal : {} {}".format(cal_points.shape,cal_gaze.shape))
    # print("val : {} {}".format(val_points.shape,val_gaze.shape))
    # # Todo: fix the issue regarding the opencv installation
    # # cv2.destroyAllWindows()
    # cap.release()
    # vid.close()

    return True# imgpoints, image_list, marker_found_index#val_points, val_gaze, cal_points, cal_gaze
