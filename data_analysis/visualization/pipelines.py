# Calibration script
import vm_preproc as vmp
import vedb_store
import numpy as np
import tqdm
import os
from ..gaze import gaze_utils
from .gaze_quality import plot_gaze_accuracy_heatmap
import cv2
import imageio
import itertools

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

def show_world_v01(
    session_directory,
    session_folder,
    param_dict,
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
    session_id = session_folder[-19:] + '/'
    output_path =   param_dict['directory']['saving_directory'] + session_id
    processed_path = param_dict['directory']['gaze_directory'] + session_id
    # Deal with inputs
    if output_path is None:
        #output_path = session_folder
        raise ValueError("parameters' yaml file doesn't have valid saving_directory!")
    else:
        print("saving results to: ", output_path)

    start_time = param_dict['visualization']['start_time']
    end_time = param_dict['visualization']['end_time']
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
    session = vedb_store.Session(folder=session_folder)
    # (1) Read Start and End Indexes
    start_index = (start_time[0] * 60 + start_time[1]) * fps
    end_index = (end_time[0] * 60 + end_time[1]) * fps
    # (2) Read Gaze File
    gaze_file = processed_path + '/' + gaze_calibration_tag + '_gaze.npz'

    if os.path.exists(gaze_file):
        print("Loading gaze data")
        gaze_arrays = {}
        dat = np.load(gaze_file, allow_pickle=True)
        for k in dat.keys():
            gaze_arrays[k] = dat[k]
        gaze_list = gaze_utils.arraydict_to_dictlist(gaze_arrays)
    else:
        raise ValueError("No valid gaze file was found!", gaze_file)
    for key, value in gaze_list[0].items():
        print(key, value)

    # (3) Read Video File
    world_video_file = param_dict['directory']['session_directory'] + session_id + "world.mp4"
    print("world Video File: ", world_video_file)
    vid = imageio.get_reader(world_video_file, "ffmpeg")
    cap = cv2.VideoCapture(world_video_file)
    world_time_stamp_file = param_dict['directory']['session_directory'] + session_id + "world_timestamps.npy"
    world_time_stamp = np.load(world_time_stamp_file)
    gaze_timestamp = []
    for my_index in range(len(gaze_list)):
        gaze_timestamp.append(gaze_list[my_index]['gaze_binocular']['timestamp'])


    # (4) Read Calibration Marker File
    reference_file = processed_path + '/' + gaze_calibration_tag + '_calibration_ref_pos.npz'
    if os.path.exists(reference_file):
        print("Loading reference data: Calibration")
        cal_reference_arrays = {}
        dat = np.load(reference_file, allow_pickle=True)
        for k in dat.keys():
            cal_reference_arrays[k] = dat[k]
        reference_list = gaze_utils.arraydict_to_dictlist(cal_reference_arrays)
    else:
        raise ValueError("No valid reference/calibration file was found!", reference_file)
    for key, value in reference_list[0].items():
        print(key, value)
    cal_reference_timestamps = cal_reference_arrays['timestamp']

    # (5) Read Validation Marker File
    reference_file = processed_path + '/' + gaze_calibration_tag + '_validation_ref_pos.npz'
    if os.path.exists(reference_file):
        print("Loading reference data: Validation")
        val_reference_arrays = {}
        dat = np.load(reference_file, allow_pickle=True)
        for k in dat.keys():
            val_reference_arrays[k] = dat[k]
        reference_list = gaze_utils.arraydict_to_dictlist(val_reference_arrays)
    else:
        raise ValueError("No valid reference/validation file was found!", reference_file)
    for key, value in reference_list[0].items():
        print(key, value)
    val_pos, val_reference_timestamps = clean_up_reference_dict(val_reference_arrays)

    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * world_scale)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * world_scale)
    video_size = (frame_width, frame_height)
    print("frame size:", video_size)
    print("First Frame = %d" % start_index)
    print("Last Frame = %d" % end_index)
    print("scale[x,y] = ", world_scale)

    # Instantiate the video recorder in order to store the processed images to an output video
    if save_video:
        fourcc = 'mp4v'
        output_video_file = os.path.join(output_path, str(tag)) +".mp4"
        print("output video file: ", output_video_file)
        out_video = cv2.VideoWriter(output_video_file,cv2.VideoWriter_fourcc(*fourcc), fps, video_size)

    # Read the next frame from the video.
    for i in range(start_index, end_index):

        img = vid.get_data(i)
        img[:, :, [0, 2]] = img[:, :, [2, 0]]
        img = cv2.resize(img, None, fx=world_scale, fy=world_scale)
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gaze_index = np.argmin(np.abs((gaze_timestamp - world_time_stamp[i]).astype(float)))
        gaze_norm_x = gaze_list[gaze_index]['gaze_binocular']['norm_pos'][0]
        gaze_norm_y = gaze_list[gaze_index]['gaze_binocular']['norm_pos'][1]

        gaze_pixel_x = int(gaze_norm_x * frame_width)
        gaze_pixel_y = int((gaze_norm_y) * frame_height)
        #frame_no_gaze = img.copy()
        if (min(np.abs(val_reference_timestamps - world_time_stamp[i]))<0.03 and len(val_reference_timestamps)>0):
            reference_index = np.argmin(np.abs(val_reference_timestamps - world_time_stamp[i]))
            # ref_pixel_x = int(cal_reference_arrays['location'][reference_index, 0] * world_scale)
            # ref_pixel_y = int(cal_reference_arrays['location'][reference_index, 1] * world_scale)
            val_pos[reference_index, :, 0] = val_pos[reference_index, :, 0] * frame_width * (2)
            val_pos[reference_index, :, 1] = val_pos[reference_index, :, 1] * frame_height * (1)
            ref_corners = val_pos[reference_index, :, :]
            img = cv2.drawChessboardCorners(img, (6, 8), ref_corners, True)

        img = cv2.circle(img, (gaze_pixel_x, gaze_pixel_y), 6, (255, 255, 0), 3)

        if (min(np.abs(cal_reference_timestamps - world_time_stamp[i]))<0.03):
            reference_index = np.argmin(np.abs(cal_reference_timestamps - world_time_stamp[i]))
            ref_pixel_x = int(cal_reference_arrays['location'][reference_index, 0] * world_scale)
            ref_pixel_y = int(cal_reference_arrays['location'][reference_index, 1] * world_scale)
            img = cv2.circle(img, (ref_pixel_x, ref_pixel_y), 8, (200, 25, 205), 2)
        # Todo: Create a separate function for adding text info on the frame
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (10, 30)
        # fontScale
        font_scale = 0.5
        # Blue color in BGR
        color = (0, 255, 255)
        # Line thickness of 2 px
        thickness = 1
        confidence = "{0:.2f}".format(gaze_list[i]['gaze_binocular']['confidence'])
        text = "confidence: " + confidence
        # Using cv2.putText() method
        img = cv2.putText(img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
        #
        # w = 50
        # h = 50
        # x_min = gaze_pixel_x - w
        # x_max = gaze_pixel_x + w
        #
        # y_min = gaze_pixel_y - h
        # y_max = gaze_pixel_y + h
        #
        # if gaze_pixel_x - w < 0:
        #     x_min = 0
        #     x_max = 2 * w
        # if gaze_pixel_x + w >= (horizontal_pixels * scale_x):
        #     x_min = horizontal_pixels * scale_x - 2 * w - 1
        #     x_max = horizontal_pixels * scale_x - 1
        #
        # if gaze_pixel_y - h < 0:
        #     y_min = 0
        #     y_max = 2 * h
        # if gaze_pixel_y + h >= (vertical_pixels * scale_y):
        #     y_min = vertical_pixels * scale_y - 2 * h - 1
        #     y_max = vertical_pixels * scale_y - 1
        #
        # range_x = np.arange(int(x_min), int(x_max))
        # range_y = np.arange(int(y_min), int(y_max))
        #
        # fovea = frame_no_gaze[int(y_min): int(y_max), int(x_min): int(x_max), :]
        # print(frame_no_gaze.shape, fovea.shape)
        # print(range_x.shape, range_y.shape)
        # print(range_x.min(), range_x.max(), range_y.min(), range_y.max())
        cv2.imshow("img", img)
        if save_video:
            out_video.write(img)
        #cv2.imshow("fovea", fovea)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print("\nDone!")

    cv2.destroyAllWindows()
    cap.release()
    vid.close()
    # Release the video writer handler so that the output video is saved to disk
    if save_video:
        out_video.release()

    return True#final_result

def show_world_v02(
    session_directory,
    session_folder,
    param_dict,
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
    session_id = session_folder[-19:] + '/'
    output_path =   param_dict['directory']['saving_directory'] + session_id
    processed_path = param_dict['directory']['gaze_directory'] + session_id
    # Deal with inputs
    if output_path is None:
        #output_path = session_folder
        raise ValueError("parameters' yaml file doesn't have valid saving_directory!")
    else:
        print("saving results to: ", output_path)

    start_time = param_dict['visualization']['start_time']
    end_time = param_dict['visualization']['end_time']
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
    session = vedb_store.Session(folder=session_folder)
    # (1) Read Start and End Indexes
    start_index = (start_time[0] * 60 + start_time[1]) * fps
    end_index = (end_time[0] * 60 + end_time[1]) * fps
    # (2) Read Gaze File
    gaze_file = processed_path + '/' + gaze_calibration_tag + '_gaze.npz'

    if os.path.exists(gaze_file):
        print("Loading gaze data")
        gaze_arrays = {}
        dat = np.load(gaze_file, allow_pickle=True)
        for k in dat.keys():
            gaze_arrays[k] = dat[k]
        gaze_list = gaze_utils.arraydict_to_dictlist(gaze_arrays)
    else:
        raise ValueError("No valid gaze file was found!", gaze_file)
    for key, value in gaze_list[0].items():
        print(key, value)

    # (3) Read Video File
    world_video_file = param_dict['directory']['session_directory'] + session_id + "world.mp4"
    print("world Video File: ", world_video_file)
    vid = imageio.get_reader(world_video_file, "ffmpeg")
    cap = cv2.VideoCapture(world_video_file)
    world_time_stamp_file = param_dict['directory']['session_directory'] + session_id + "world_timestamps.npy"
    world_time_stamp = np.load(world_time_stamp_file)
    gaze_timestamp = []
    for my_index in range(len(gaze_list)):
        gaze_timestamp.append(gaze_list[my_index]['gaze_binocular']['timestamp'])


    # (4) Read Calibration Marker File
    reference_file = processed_path + '/' + gaze_calibration_tag + '_calibration_ref_pos.npz'
    if os.path.exists(reference_file):
        print("Loading reference data: Calibration")
        cal_reference_arrays = {}
        dat = np.load(reference_file, allow_pickle=True)
        for k in dat.keys():
            cal_reference_arrays[k] = dat[k]
        reference_list = gaze_utils.arraydict_to_dictlist(cal_reference_arrays)
    else:
        raise ValueError("No valid reference/calibration file was found!", reference_file)
    for key, value in reference_list[0].items():
        print(key, value)
    cal_reference_timestamps = cal_reference_arrays['timestamp']

    # (5) Read Validation Marker File
    reference_file = processed_path + '/' + gaze_calibration_tag + '_validation_ref_pos.npz'
    if os.path.exists(reference_file):
        print("Loading reference data: Validation")
        val_reference_arrays = {}
        dat = np.load(reference_file, allow_pickle=True)
        for k in dat.keys():
            val_reference_arrays[k] = dat[k]
        reference_list = gaze_utils.arraydict_to_dictlist(val_reference_arrays)
    else:
        raise ValueError("No valid reference/validation file was found!", reference_file)
    for key, value in reference_list[0].items():
        print(key, value)
    val_pos, val_reference_timestamps = clean_up_reference_dict(val_reference_arrays)

    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * world_scale)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * world_scale)
    video_size = (frame_width, frame_height)
    print("frame size:", video_size)
    print("First Frame = %d" % start_index)
    print("Last Frame = %d" % end_index)
    print("scale[x,y] = ", world_scale)

    # Instantiate the video recorder in order to store the processed images to an output video
    # if save_video:
    #     fourcc = 'mp4v'
    #     output_video_file = os.path.join(output_path, str(tag)) +".mp4"
    #     print("output video file: ", output_video_file)
    #     out_video = cv2.VideoWriter(output_video_file,cv2.VideoWriter_fourcc(*fourcc), fps, video_size)
    cal_points = []
    cal_gaze = []
    val_points = []
    val_gaze = []
    # Read the next frame from the video.
    for i in range(start_index, end_index):

        print("frame index= ", i ,end='\r')
        gaze_index = np.argmin(np.abs((gaze_timestamp - world_time_stamp[i]).astype(float)))
        gaze_norm_x = gaze_list[gaze_index]['gaze_binocular']['norm_pos'][0]
        gaze_norm_y = gaze_list[gaze_index]['gaze_binocular']['norm_pos'][1]

        gaze_pixel_x = int(gaze_norm_x * frame_width)
        gaze_pixel_y = int((gaze_norm_y) * frame_height)
        #frame_no_gaze = img.copy()
        if len(val_reference_timestamps)>0:
            if (min(np.abs(val_reference_timestamps - world_time_stamp[i]))<0.03):
                reference_index = np.argmin(np.abs(val_reference_timestamps - world_time_stamp[i]))
                # ref_pixel_x = int(cal_reference_arrays['location'][reference_index, 0] * world_scale)
                # ref_pixel_y = int(cal_reference_arrays['location'][reference_index, 1] * world_scale)
                val_pos[reference_index, :, 0] = val_pos[reference_index, :, 0] * frame_width * (2)
                val_pos[reference_index, :, 1] = val_pos[reference_index, :, 1] * frame_height * (1)
                ref_corners = val_pos[reference_index, :, :]
                p = val_pos[reference_index, :]
                p[:,0] = p[:,0]*2
                val_points.append(np.mean(p,axis=0))
                val_gaze.append([gaze_norm_x,gaze_norm_y])


        if (min(np.abs(cal_reference_timestamps - world_time_stamp[i]))<0.03):
            reference_index = np.argmin(np.abs(cal_reference_timestamps - world_time_stamp[i]))
            ref_pixel_x = int(cal_reference_arrays['location'][reference_index, 0] * world_scale)
            ref_pixel_y = int(cal_reference_arrays['location'][reference_index, 1] * world_scale)
            cal_points.append([cal_reference_arrays['location'][reference_index, 0], cal_reference_arrays['location'][reference_index, 1]])
            cal_gaze.append([gaze_norm_x,gaze_norm_y])

    val_points = np.array(val_points)
    val_gaze = np.array(val_gaze)
    cal_points = np.array(cal_points)
    cal_gaze = np.array(cal_gaze)
    print("\nDone!")
    print("cal : {} {}".format(cal_points.shape,cal_gaze.shape))
    print("val : {} {}".format(val_points.shape,val_gaze.shape))
    cv2.destroyAllWindows()
    cap.release()
    vid.close()

    return val_points, val_gaze, cal_points, cal_gaze


def show_gaze_accuracy_v01(
    session_directory,
    session_folder,
    param_dict,
    world_scale=1,
    progress_bar=tqdm.tqdm,
):
    print(param_dict.keys())
    # Todo: Read length of the session id from parameters?
    session_id = session_folder[-19:] + '/'
    output_path = param_dict['directory']['saving_directory'] + session_id
    gaze_direcory = param_dict['directory']['gaze_directory'] + session_id

    # Deal with inputs
    if output_path is None:
        #output_path = session_folder
        raise ValueError("parameters' yaml file doesn't have valid saving_directory!")
    else:
        print("saving results to: ", output_path)

    start_time = param_dict['visualization']['start_time']
    end_time = param_dict['visualization']['end_time']
    save_video = param_dict['visualization']['save_world_output']
    world_scale = param_dict['visualization']['world_scale']
    eye_scale = param_dict['visualization']['eye_scale']

    gaze_calibration_tag = param_dict['calibration']['pupil_detection'] + '_' +\
          param_dict['calibration']['eye'] + '_' +\
          param_dict['calibration']['algorithm']

    # Todo: Read this from session info
    fps = 30
    if not os.path.exists(output_path):
        print("creating", output_path)
        os.makedirs(output_path)

    tag = param_dict['calibration']['pupil_detection'] + '_' +\
          param_dict['calibration']['eye'] + '_' +\
          param_dict['calibration']['algorithm']
    print('tag : ', tag)
    string_name = os.path.join(gaze_direcory, tag + "_{step}.npz")
    print("file_name", string_name)

    # (0) Get session
    session = vedb_store.Session(folder=session_folder)
    # (1) Read Start and End Indexes
    start_index = (start_time[0] * 60 + start_time[1]) * fps
    end_index = (end_time[0] * 60 + end_time[1]) * fps
    # (2) Read Gaze File
    gaze_file = string_name.format(step="gaze")
    print("gaze_file: ", gaze_file)

    if os.path.exists(gaze_file):
        print("Loading gaze data")
        gaze_arrays = {}
        dat = np.load(gaze_file, allow_pickle=True)
        for k in dat.keys():
            gaze_arrays[k] = dat[k]
        gaze_list = gaze_utils.arraydict_to_dictlist(gaze_arrays)
        gaze_timestamp = []
        for my_index in range(len(gaze_list)):
            gaze_timestamp.append(gaze_list[my_index]['gaze_binocular']['timestamp'])

    else:
        raise ValueError("No valid gaze file was found!", gaze_file)
    for key, value in gaze_list[0].items():
        print("Gaze List", key, value)

    # (3) Calibration Marker detection
    cal_ref_file = string_name.format(step="calibration_ref_pos")

    if os.path.exists(cal_ref_file):
        print("Loading calibration markers")
        ref_arrays = {}
        data = np.load(cal_ref_file, allow_pickle=True)
        for k in data.keys():
            ref_arrays[k] = data[k]
        cal_ref_list = gaze_utils.arraydict_to_dictlist(ref_arrays)
        for key, value in cal_ref_list[0].items():
            print("Calibration List", key, value)
    else:
        print("No valid calibration reference file", cal_ref_file)
    print("calibration len: ", len(cal_ref_list))
    gaze_norm_x = []
    gaze_norm_y = []
    confidence = []
    print("Finding gaze indexes: ", len(ref_arrays['timestamp']))
    for ref_time in ref_arrays['timestamp']:
        gaze_index = np.argmin(np.abs((gaze_timestamp - ref_time).astype(float)))
        gaze_norm_x.append(gaze_list[gaze_index]['gaze_binocular']['norm_pos'][0])
        gaze_norm_y.append(gaze_list[gaze_index]['gaze_binocular']['norm_pos'][1])
        confidence.append(gaze_list[gaze_index]['gaze_binocular']['confidence'])
    gaze_array = np.array([gaze_norm_x, gaze_norm_y])
    print(gaze_array.shape)
    file_name = os.path.join(output_path, tag + "_{step}.png")
    file_name.format(step="calibration_accuracy")
    plot_gaze_accuracy_heatmap(ref_arrays['norm_pos'], gaze_array.T, confidence, file_name, reference_type="calibration")


    # (4) Validation Marker detection
    # val_ref_file = string_name.format(step="validation_ref_pos_dict")

    # if os.path.exists(val_ref_file):
    #     print("Loading validation markers")
    #     ref_arrays = {}
    #     data = np.load(val_ref_file, allow_pickle=True)
    #     for k in data.keys():
    #         ref_arrays[k] = data[k]
    #     val_ref_list = gaze_utils.arraydict_to_dictlist(ref_arrays)
    #     for key, value in val_ref_list[0].items():
    #         print("Validation List", key, value)
    # else:
    #     print("No valid validation reference file", val_ref_file)
    # print("validation len: ", len(val_ref_list))
    #
    # gaze_norm_x = []
    # gaze_norm_y = []
    # ref_x = []
    # ref_y = []
    # confidence = None
    # # for key, value in ref_arrays.items():
    # #     print("ref array", key, value)
    # print("Finding gaze indexes: ", len(ref_arrays[0]['timestamp']))
    # count = 0
    # for myDict in val_ref_list:
    #     if(len()>0)
    #     gaze_index = np.argmin(np.abs((gaze_timestamp - myDict['timestamp']).astype(float)))
    #     gaze_norm_x.append(gaze_list[gaze_index]['gaze_binocular']['norm_pos'][0])
    #     gaze_norm_y.append(gaze_list[gaze_index]['gaze_binocular']['norm_pos'][1])
    #     ref_x.append(ref_arrays[ref_time]['mean_norm_pos'][0])
    #     ref_y.append(ref_arrays[ref_time]['mean_norm_pos'][1])
    # gaze_array = np.array([gaze_norm_x, gaze_norm_y])
    # print(gaze_array.shape)
    # file_name = os.path.join(output_path, tag + "_{step}.png")
    # file_name.format(step="validation_accuracy")
    # ref_array = np.array([ref_x, ref_y])
    # plot_gaze_accuracy_heatmap(ref_array.T, gaze_array.T, confidence, file_name, reference_type="validation")

    return


def show_eye_confidence_v01(
    session_directory,
    session_folder,
    param_dict,
    string_name = None,
    world_scale=1,
    progress_bar=tqdm.tqdm,
):
    import matplotlib.pyplot as plt
    print(param_dict.keys())
    # Todo: Read length of the session id from parameters?
    session_id = session_folder[-19:] + '/'
    output_path = param_dict['directory']['saving_directory'] + session_id
    gaze_path = param_dict['directory']['gaze_directory'] + session_id
    # Deal with inputs
    if output_path is None:
        #output_path = session_folder
        raise ValueError("parameters' yaml file doesn't have valid saving_directory!")
    else:
        print("saving results to: ", output_path)

    start_time = param_dict['visualization']['start_time']
    end_time = param_dict['visualization']['end_time']
    save_video = param_dict['visualization']['save_eye_output']
    scale = param_dict['visualization']['eye_scale']
    eye_index_dict = {"right": [0], "left": [1], "binocular": [0, 1]}
    eye_string = {1: "right", 0: "left"}
    eye_index = eye_index_dict[param_dict['visualization']['eye_index']]

    gaze_calibration_tag = param_dict['calibration']['pupil_detection'] + '_' +\
          param_dict['calibration']['eye'] + '_' +\
          param_dict['calibration']['algorithm']

    tag = param_dict['calibration']['pupil_detection'] + '_' +\
          param_dict['calibration']['eye'] + '_' +\
          param_dict['calibration']['algorithm']
    print('tag : ', tag)
    pupil_tag = param_dict['calibration']['pupil_detection'] + '_' +\
          param_dict['calibration']['eye'] + '_' +\
          param_dict['calibration']['algorithm']

    # Todo: Read this from session info
    skip_frame = int(param_dict['visualization']['eye_fps_scale'])
    print("Skip Frame", skip_frame)
    fps = int(120/skip_frame)
    print('tag : ', tag)
    if string_name is None:
        string_name = os.path.join(gaze_path, pupil_tag + "_{step}.npz")
    print("file_name", string_name)
    if not os.path.exists(output_path):
        print("creating", output_path)
        os.makedirs(output_path)
    # (0) Get session
    session = vedb_store.Session(folder=session_folder)
    # (1) Read Start and End Indexes
    start_index = (start_time[0] * 60 + start_time[1]) * fps
    end_index = (end_time[0] * 60 + end_time[1]) * fps
    # (2) Read Gaze File
    gaze_file = session_folder + '/' + gaze_calibration_tag + '_gaze.npz'

    if os.path.exists(gaze_file):
        print("Loading gaze data")
        gaze_arrays = {}
        dat = np.load(gaze_file, allow_pickle=True)
        for k in dat.keys():
            gaze_arrays[k] = dat[k]
        gaze_list = gaze_utils.arraydict_to_dictlist(gaze_arrays)
    else:
        raise ValueError("No valid gaze file was found!", gaze_file)
    for key, value in gaze_list[0].items():
        print(key, value)

    confidence = []
    for i in range(len(gaze_list)):
        confidence.append(gaze_list[i]['gaze_binocular']['confidence'])
    plt.figure(figsize=(12, 10))

    # plt.boxplot(diff_r_az, notch=True, showfliers=False)
    boxprops = dict(linestyle='-', linewidth=3)
    whiskerprops = dict(linestyle='-', linewidth=3)
    capprops = dict(linestyle='-', linewidth=3)
    plt.boxplot([confidence], boxprops=boxprops, whiskerprops=whiskerprops,
                capprops=capprops, notch = True, showfliers = False)
    plt.title("Gaze Confidence", fontsize=16)
    plt.ylabel("Confidence", fontsize=16)
    # plt.xticks([1, 2, 3, 4], ["azimuth_right", "elevation_right", "azimuth_left", "elevation_left"], fontsize=18)
    plt.xticks([1], ["gaze_2D"], fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    file_name = os.path.join(output_path, gaze_calibration_tag + "_{step}.png")
    file_name.format(step="gaze_confidence")
    print(file_name)
    plt.savefig(file_name, dpi=150, transparent=False)
    #plt.show()
    plt.close()

    return True

def show_eye_v01(
    session_directory,
    session_folder,
    param_dict,
    string_name=None,
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
    session_id = session_folder[-19:] + '/'
    output_path = param_dict['directory']['saving_directory'] + session_id
    gaze_path = param_dict['directory']['gaze_directory'] + session_id
    # Deal with inputs
    if output_path is None:
        #output_path = session_folder
        raise ValueError("parameters' yaml file doesn't have valid saving_directory!")
    else:
        print("saving results to: ", output_path)

    start_time = param_dict['visualization']['start_time']
    end_time = param_dict['visualization']['end_time']
    save_video = param_dict['visualization']['save_eye_output']
    scale = param_dict['visualization']['eye_scale']
    eye_index_dict = {"right": [0], "left": [1], "binocular": [0, 1]}
    eye_string = {1: "right", 0: "left"}
    eye_index = eye_index_dict[param_dict['visualization']['eye_index']]

    gaze_calibration_tag = param_dict['calibration']['pupil_detection'] + '_' +\
          param_dict['calibration']['eye'] + '_' +\
          param_dict['calibration']['algorithm']

    tag = 'eye_' +\
          param_dict['visualization']['eye_index'] + '_' +\
          str(start_time[0])+ '_' +\
          str(start_time[1]) + '_' +\
          str(end_time[0])+ '_' +\
          str(end_time[1])
    pupil_tag = param_dict['calibration']['pupil_detection'] + '_' +\
          param_dict['calibration']['eye'] + '_' +\
          param_dict['calibration']['algorithm']

    # Todo: Read this from session info
    skip_frame = int(param_dict['visualization']['eye_fps_scale'])
    print("Skip Frame", skip_frame)
    fps = int(120/skip_frame)
    print('tag : ', tag)
    if string_name is None:
        string_name = os.path.join(gaze_path, pupil_tag + "_{step}.npz")
    print("file_name", string_name)
    if not os.path.exists(output_path):
        print("creating", output_path)
        os.makedirs(output_path)
    # (0) Get session
    session = vedb_store.Session(folder=session_folder)
    # (1) Read Start and End Indexes
    start_index = (start_time[0] * 60 + start_time[1]) * fps
    end_index = (end_time[0] * 60 + end_time[1]) * fps
    # (2) Read Gaze File
    gaze_file = session_folder + '/' + gaze_calibration_tag + '_gaze.npz'

    if os.path.exists(gaze_file):
        print("Loading gaze data")
        gaze_arrays = {}
        dat = np.load(gaze_file, allow_pickle=True)
        for k in dat.keys():
            gaze_arrays[k] = dat[k]
        gaze_list = gaze.gaze_utils.arraydict_to_dictlist(gaze_arrays)
    else:
        raise ValueError("No valid gaze file was found!", gaze_file)
    for key, value in gaze_list[0].items():
        print(key, value)

    # (2) Read Video File(s)
    # if eye_index is not 2:
    #     eye_video_file = param_dict['directory']['session_directory'] + session_id + "eye" + str(eye_index) + ".mp4"
    #     print("Eye Video File: ", eye_video_file)
    #     vid = imageio.get_reader(eye_video_file, "ffmpeg")
    #     cap = cv2.VideoCapture(eye_video_file)
    #     eye_time_stamp_file = param_dict['directory']['session_directory'] + session_id + "eye" + str(eye_index) + "_timestamps.npy"
    #     eye_time_stamp = np.load(eye_time_stamp_file)
    #     gaze_timestamp = []
    #     for my_index in range(len(gaze_list)):
    #         gaze_timestamp.append(gaze_list[my_index]['gaze_binocular']['timestamp'])
    #
    # else:
    vid = []
    cap = []
    gaze_timestamp = []
    pupil_dicts = []
    count = 0
    for i in eye_index:
        eye_video_file = param_dict['directory']['session_directory'] + session_id + "eye" + str(i) + ".mp4"
        print("Eye Video File: ", eye_video_file)
        vid.append(imageio.get_reader(eye_video_file, "ffmpeg"))
        cap.append(cv2.VideoCapture(eye_video_file))
        eye_time_stamp_file = param_dict['directory']['session_directory'] + session_id + "eye" + str(i) + "_timestamps.npy"
        _gaze_timestamp = np.load(eye_time_stamp_file)
        gaze_timestamp.append(_gaze_timestamp)
        frame_height = int(cap[count].get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
        frame_width = int(cap[count].get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
        step = "pupil_pos_" + eye_string[i]
        pupil_file_name = string_name.format(step=step)
        print(pupil_file_name)
        pupil_dicts.append(np.load(pupil_file_name, allow_pickle=True))
        print("File includes: ", pupil_dicts[count].files)
        count = count + 1
    print("total number of dicts: ", len(pupil_dicts))
    print("gaze timestamp shape: ", np.asarray(gaze_timestamp).shape)
    print("number of videos: ", np.asarray(cap).shape)
    print(param_dict['visualization']['eye_index'], eye_index)
    video_size = (frame_width*len(eye_index), frame_height)
    print("frame size:", video_size)

    print("First Frame = %d" % start_index)
    print("Last Frame = %d" % end_index)
    print("scale[x,y] = ", scale)

    # Instantiate the video recorder in order to store the processed images to an output video
    if save_video:
        fourcc = 'mp4v'
        output_video_file = os.path.join(output_path, str(tag)) +".mp4"
        print("output video file: ", output_video_file)
        out_video = cv2.VideoWriter(output_video_file, cv2.VideoWriter_fourcc(*fourcc), fps, video_size)
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org_1 = (int(20*scale), int(30*scale))
    org_2 = (int(20*scale), int(60*scale))
    # fontScale
    fontScale = 1 * scale
    # Blue color in BGR
    color = (250, 250, 0)
    # Line thickness of 2 px
    thickness = 1
    # Window name in which image is displayed
    window_name = 'Image'
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Blue color in BGR
    font_color = (255, 250, 250)

    # Read the next frame from the video.
    for i in range(start_index, end_index, skip_frame):
        final_frame = []
        frame = []
        for j in eye_index:
            img = vid[j].get_data(i)
            img[:, :, [0, 2]] = img[:, :, [2, 0]]
            img = cv2.resize(img, None, fx=scale, fy=scale)
            center_coordinates = (scale * pupil_dicts[j]['location'][i, 0], scale * pupil_dicts[j]['location'][i, 1])
            axesLength = (scale * pupil_dicts[j]['ellipse'][i]['axes'][0], scale * pupil_dicts[j]['ellipse'][i]['axes'][1])
            angle = pupil_dicts[j]['ellipse'][i]['angle']
            startAngle = 0
            endAngle = 360
            # Using cv2.ellipse() method
            # Draw a ellipse with blue line borders of thickness of -1 px
            img = cv2.ellipse(img, (center_coordinates, axesLength, angle), color, thickness)
            img = cv2.putText(img, "Frame#:" + str(i), org_1, font,
                                fontScale, font_color, 1, cv2.LINE_AA)
            img = cv2.putText(img, 'Confidence:' + "{0:.2f}".format(pupil_dicts[j]['confidence'][i]), org_2, font,
                                fontScale, font_color, 1, cv2.LINE_AA)
            frame.append(img)
        if len(eye_index) > 1:
            final_frame = np.concatenate((frame[0], frame[1]), axis=1)
        else:
            final_frame = frame
        cv2.imshow("Pupil ellipse", final_frame)
        # cv.imshow("motion histogram",img)
        if save_video:
            out_video.write(final_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print("\nDone!")

    cv2.destroyAllWindows()
    for i in range(len(eye_index)):
        cap[i].release()
        vid[i].close()
    # Release the video writer handler so that the output video is saved to disk
    if save_video:
        out_video.release()

    return True#final_result
