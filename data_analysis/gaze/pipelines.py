# Calibration script
import vm_preproc as vmp
import vedb_store
import numpy as np
import tqdm
import os
from . import pupil_detection_pl, calibrate_pl
from . import marker_detection, gaze_utils
from pupil_recording_interface.externals.gaze_mappers import Binocular_Gaze_Mapper

def pupil_2d_monocular_v01(
    session_folder,
    sname=None,  # Base for each file?
    tag="pupil_2d_monocular_v01",
    output_path=None,
    batch_size_pupil="auto",
    batch_size_marker="auto",
    marker_rescale=0.5,
    progress_bar=tqdm.tqdm,
    properties=None,
):
    """
    Parameters
    ----------
    tag : A short label for the pipeline
    session_folder : string
        file path to session. Ultimately want to replace this with a session
        object from the database.
    sname : format string
        must contain {step}; if provided, tag and output_path are ignored

    Notes
    -----
    Ultimately, for saving files, we want the output of each step saved
    along with the function and parameters that were used to generate it.

    This is not the case now. Needs work.
    """
    # Deal with inputs
    if output_path is None:
        output_path = session_folder
    if sname is None:
        sname = os.path.join(output_path, "gaze_vedb", tag + "_{step}.npz")
    fdir, _ = os.path.split(sname)
    if not os.path.exists(fdir):
        print("creating", fdir)
        os.makedirs(fdir)
    # (0) Get session
    ses = vedb_store.Session(folder=session_folder)
    # (1) Pupil detection (L, R)
    fn_pupil = pupil_detection_pl.plabs_detect_pupil

    pupil_file_left = sname.format(step="pupilpos_left")
    if os.path.exists(pupil_file_left):
        print("Loading pupils left")
        pupil_arrays_left = {}
        dat = np.load(pupil_file_left, allow_pickle=True)
        for k in dat.keys():
            pupil_arrays_left[k] = dat[k]
        pupil_list_left = gaze_utils.arraydict_to_dictlist(pupil_arrays_left)
    else:
        # Get eye files (Left eye)
        eye_left_time_file, eye_left_video_file = ses.paths["eye_left"]
        inputs_pupil_left = dict(
            fpaths=dict(eye_video=eye_left_video_file, timestamps=eye_left_time_file,),
            variable_names=None,
        )
        print("\n\nRunning pupil detection for the left eye\n\n")
        # Run pupil detection
        pupil_list_left = vmp.utils.batch_run(
            fn_pupil,
            inputs_pupil_left,
            batch_size=batch_size_pupil,
            batch_combine_fn=vmp.utils.list_reduce,
            progress_bar=progress_bar,
            id=1,  # left eye # FIX ME: OPTION HERE FOR R, L, or binocular
            properties=properties,
        )
        # Get arrays instead of list of dicts
        pupil_arrays_left = gaze_utils.dictlist_to_arraydict(pupil_list_left)
        # Save pupil detection
        pupil_file_left = sname.format(step="pupilpos_left")
        np.savez(pupil_file_left, **pupil_arrays_left)

    pupil_file_right = sname.format(step="pupilpos_right")
    if os.path.exists(pupil_file_right):
        print("Loading pupils right")
        pupil_arrays_right = {}
        dat = np.load(pupil_file_right, allow_pickle=True)
        for k in dat.keys():
            pupil_arrays_right[k] = dat[k]
        pupil_list_right = gaze_utils.arraydict_to_dictlist(pupil_arrays_right)
    else:
        # Get eye files (Right eye)
        eye_right_time_file, eye_right_video_file = ses.paths["eye_right"]
        inputs_pupil_right = dict(
            fpaths=dict(
                eye_video=eye_right_video_file, timestamps=eye_right_time_file,
            ),
            variable_names=None,
        )
        print("\n\nRunning pupil detection for the right eye\n\n")
        # Run pupil detection
        pupil_list_right = vmp.utils.batch_run(
            fn_pupil,
            inputs_pupil_right,
            batch_size=batch_size_pupil,
            batch_combine_fn=vmp.utils.list_reduce,
            progress_bar=progress_bar,
            id=0,  # right eye # FIX ME: OPTION HERE FOR R, L, or binocular
            properties=properties,
        )
        # Get arrays instead of list of dicts
        pupil_arrays_right = gaze_utils.dictlist_to_arraydict(pupil_list_right)
        # Save pupil detection

        np.savez(pupil_file_right, **pupil_arrays_right)

    # (2) Marker detection
    ref_file = sname.format(step="markerpos")
    # Get world video files
    world_time_file, world_video_file = ses.paths["world_camera"]
    # Load 1 second of data 10 seconds in (to allow time for camera to start)
    world_time, world_video = ses.load("world_camera", idx=(10, 11))
    _, video_vdim, video_hdim = world_video.shape[:3]

    if os.path.exists(ref_file):
        print("Loading markers")
        ref_arrays = {}
        dat = np.load(ref_file, allow_pickle=True)
        for k in dat.keys():
            ref_arrays[k] = dat[k]
        ref_list = gaze_utils.arraydict_to_dictlist(ref_arrays)
    else:
        fn_marker = marker_detection.find_concentric_circles
        inputs_marker = dict(
            fpaths=dict(video_data=world_video_file, timestamps=world_time_file),
            variable_names=None,
        )
        print("\n\nRunning marker detection \n\n")
        # Run marker detection
        ref_list = vmp.utils.batch_run(
            fn_marker,
            inputs_marker,
            batch_size=batch_size_marker,
            batch_combine_fn=vmp.utils.list_reduce,
            scale=marker_rescale,
            progress_bar=progress_bar,
        )
        # Get arrays instead of dicts
        ref_arrays = gaze_utils.dictlist_to_arraydict(ref_list)
        # Save calibration markers
        np.savez(ref_file, **ref_arrays)

    # (3) Calibrate
    # Get data for pupil calibration
    print("\n\nGetting data for calibration \n\n")
    is_binocular, matched_data_left = calibrate_pl.get_data(pupil_list_left, ref_list)
    # Run calibration
    # NOTE: zero index for matched_data here is because this is simply monocular,
    # and matched data only returns a 1-long tuple. If we want binocular, this will
    # need changing.
    print("\n\nRunning 2d monocular calibration [left eye] \n\n")
    method, result_left = calibrate_pl.calibrate_2d_monocular(
        matched_data_left[0], frame_size=(video_vdim, video_hdim)
    )
    # Create mapper for gaze
    cx, cy, n = result_left["args"]["params"]
    mapper_left = calibrate_pl.calibrate_2d.make_map_function(cx, cy, n)

    # (4) Map gaze to video coordinates
    # Mapper takes two inputs: normalized pupil x and y position
    print("\n\nRunning gaze mapper [left eye] \n\n")
    pupil_x, pupil_y = pupil_arrays_left["norm_pos"].T
    gaze_left = mapper_left([pupil_x, pupil_y])
    # Transpose output so time is the first dimension
    gaze_left = np.vstack(gaze_left).T

    is_binocular, matched_data_right = calibrate_pl.get_data(pupil_list_right, ref_list)
    # Run calibration
    # NOTE: zero index for matched_data here is because this is simply monocular,
    # and matched data only returns a 1-long tuple. If we want binocular, this will
    # need changing.
    print("\n\nRunning 2d monocular calibration [right eye] \n\n")
    method, result_right = calibrate_pl.calibrate_2d_monocular(
        matched_data_right[0], frame_size=(video_vdim, video_hdim)
    )
    # Create mapper for gaze
    cx, cy, n = result_right["args"]["params"]
    mapper_right = calibrate_pl.calibrate_2d.make_map_function(cx, cy, n)

    # (4) Map gaze to video coordinates
    # Mapper takes two inputs: normalized pupil x and y position
    print("\n\nRunning gaze mapper [right eye] \n\n")
    pupil_x, pupil_y = pupil_arrays_right["norm_pos"].T
    gaze_right = mapper_right([pupil_x, pupil_y])
    # Transpose output so time is the first dimension
    gaze_right = np.vstack(gaze_right).T

    gaze_file = sname.format(step="gaze")
    np.savez(gaze_file, gaze_left=gaze_left, gaze_right=gaze_right)
    return gaze_left, gaze_right

def pupil_2d_binocular_v01(
    session_folder,
    param_dict,
    string_name=None,  # Base for each file?
    tag="pupil_2d_binocular_v02",
    output_path=None,
    batch_size_pupil="auto",
    batch_size_marker="auto",
    marker_rescale=1,
    progress_bar=tqdm.tqdm,
    properties=None,
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
    output_path = param_dict['directory']['gaze_directory'] + session_id
    # Deal with inputs
    if output_path is None:
        #output_path = session_folder
        raise ValueError("parameters' yaml file doesn't have valid gaze saving_directory!")
    else:
        print("saving results to: ", output_path)

    tag = param_dict['calibration']['pupil_detection'] + '_' +\
          param_dict['calibration']['eye'] + '_' +\
          param_dict['calibration']['algorithm']
    print('tag : ', tag)
    if string_name is None:
        string_name = os.path.join(output_path, tag + "_{step}.npz")
    print("file_name", string_name)
    #fdir, _ = os.path.split(string_name)
    if not os.path.exists(output_path):
        print("creating", output_path)
        os.makedirs(output_path)
    # (0) Get session
    session = vedb_store.Session(folder=session_folder)
    # (1) Pupil detection (L, R)
    fn_pupil = pupil_detection_pl.plabs_detect_pupil

    pupil_file_left = string_name.format(step="pupil_pos_left")
    if os.path.exists(pupil_file_left):
        print("Loading pupils left")
        pupil_arrays_left = {}
        data = np.load(pupil_file_left, allow_pickle=True)
        for k in data.keys():
            pupil_arrays_left[k] = data[k]
        pupil_list_left = gaze_utils.arraydict_to_dictlist(pupil_arrays_left)
    else:
        # Get eye files (Left eye)
        eye_left_time_file, eye_left_video_file = session.paths["eye_left"]
        inputs_pupil_left = dict(
            fpaths=dict(eye_video=eye_left_video_file, timestamps=eye_left_time_file,),
            variable_names=None,
        )
        print("\n\nRunning pupil detection for the left eye\n\n")
        # Run pupil detection
        pupil_list_left = vmp.utils.batch_run(
            fn_pupil,
            inputs_pupil_left,
            batch_size=batch_size_pupil,
            batch_combine_fn=vmp.utils.list_reduce,
            progress_bar=progress_bar,
            id=1,  # left eye # FIX ME: OPTION HERE FOR R, L, or binocular
            properties=properties,
        )
        # Get arrays instead of list of dicts
        pupil_arrays_left = gaze_utils.dictlist_to_arraydict(pupil_list_left)
        # Save pupil detection
        pupil_file_left = string_name.format(step="pupil_pos_left")
        np.savez(pupil_file_left, **pupil_arrays_left)

    pupil_file_right = string_name.format(step="pupil_pos_right")
    if os.path.exists(pupil_file_right):
        print("Loading pupils right")
        pupil_arrays_right = {}
        data = np.load(pupil_file_right, allow_pickle=True)
        for k in data.keys():
            pupil_arrays_right[k] = data[k]
        pupil_list_right = gaze_utils.arraydict_to_dictlist(pupil_arrays_right)
    else:
        # Get eye files (Right eye)
        eye_right_time_file, eye_right_video_file = session.paths["eye_right"]
        inputs_pupil_right = dict(
            fpaths=dict(
                eye_video=eye_right_video_file, timestamps=eye_right_time_file,
            ),
            variable_names=None,
        )
        print("\n\nRunning pupil detection for the right eye\n\n")
        # Run pupil detection
        pupil_list_right = vmp.utils.batch_run(
            fn_pupil,
            inputs_pupil_right,
            batch_size=batch_size_pupil,
            batch_combine_fn=vmp.utils.list_reduce,
            progress_bar=progress_bar,
            id=0,  # right eye # FIX ME: OPTION HERE FOR R, L, or binocular
            properties=properties,
        )
        # Get arrays instead of list of dicts
        pupil_arrays_right = gaze_utils.dictlist_to_arraydict(pupil_list_right)
        # Save pupil detection

        np.savez(pupil_file_right, **pupil_arrays_right)

    # (2) Calibration Marker detection
    cal_ref_file = string_name.format(step="calibration_ref_pos")
    # Get world video files
    # Todo: Make sure this is loaded only if necessary
    world_time_file, world_video_file = session.paths["world_camera"]
    # Load 1 second of data 10 seconds in (to allow time for camera to start)
    world_time, world_video = session.load("world_camera", idx=(10, 11))
    _, video_vdim, video_hdim = world_video.shape[:3]

    if os.path.exists(cal_ref_file):
        print("Loading calibration markers")
        ref_arrays = {}
        data = np.load(cal_ref_file, allow_pickle=True)
        for k in data.keys():
            ref_arrays[k] = data[k]
        ref_list = gaze_utils.arraydict_to_dictlist(ref_arrays)
    else:
        fn_marker = marker_detection.find_concentric_circles
        inputs_marker = dict(
            fpaths=dict(video_data=world_video_file, timestamps=world_time_file),
            variable_names=None,
        )
        print("\n\nRunning Calibration marker detection \n\n")
        # Run marker detection
        ref_list = vmp.utils.batch_run(
            fn_marker,
            inputs_marker,
            batch_size=batch_size_marker,
            batch_combine_fn=vmp.utils.list_reduce,
            scale=marker_rescale,
            progress_bar=progress_bar,
        )
        # Get arrays instead of dicts
        ref_arrays = gaze_utils.dictlist_to_arraydict(ref_list)
        # Save calibration markers
        np.savez(cal_ref_file, **ref_arrays)

    # (3) Validation Marker detection
    val_ref_file = string_name.format(step="validation_ref_pos_dict")
    # Todo: Make sure this is handled correctly
    # Get world video files
    # world_time_file, world_video_file = session.paths["world_camera"]
    # Load 1 second of data 10 seconds in (to allow time for camera to start)
    # world_time, world_video = session.load("world_camera", idx=(10, 11))
    # _, video_vdim, video_hdim = world_video.shape[:3]

    if os.path.exists(val_ref_file):
        print("Loading validation markers")
        ref_arrays = {}
        data = np.load(val_ref_file, allow_pickle=True)
        for k in data.keys():
            ref_arrays[k] = data[k]
        ref_list = gaze_utils.arraydict_to_dictlist(ref_arrays)
    else:
        fn_marker = marker_detection.find_checkerboard
        inputs_marker = dict(
            fpaths=dict(video_data=world_video_file, timestamps=world_time_file),
            variable_names=None,
        )
        print("\n\nRunning Validation marker detection \n\n")
        # Run marker detection
        val_ref_list = vmp.utils.batch_run(
            fn_marker,
            inputs_marker,
            batch_size=batch_size_marker,
            batch_combine_fn=vmp.utils.list_reduce,
            scale=0.5,
            progress_bar=progress_bar,
        )
        print(val_ref_list)
        np.savez(val_ref_file, val_ref_list)
        val_ref_file = string_name.format(step="validation_ref_pos")
        # Get arrays instead of dicts
        val_ref_arrays = gaze_utils.dictlist_to_arraydict(val_ref_list)
        # Save calibration markers
        np.savez(val_ref_file, **val_ref_arrays)

    # (4) Append left and right pupil lists
    # Append the two pupil lists (list of dicts compatible with pupil notation)
    # And then pass the appended list to the calibration routine

    pupil_list_binocular = []
    pupil_list_binocular.extend(pupil_list_left)
    pupil_list_binocular.extend(pupil_list_right)

    # Get arrays instead of list of dicts
    pupil_arrays_binocular = gaze_utils.dictlist_to_arraydict(pupil_list_binocular)
    pupil_arrays_right = gaze_utils.dictlist_to_arraydict(pupil_list_right)
    pupil_arrays_left = gaze_utils.dictlist_to_arraydict(pupil_list_left)

    print("\n\nAppending left and right pupil positions")
    print("left:{} right:{} binocular:{} \n\n".format(len(pupil_list_left), len(pupil_list_right), len(pupil_list_binocular)))

    # (5) Calibrate
    # Get data for pupil calibration
    print("\n\nGetting data for calibration \n\n")
    is_binocular, matched_data_binocular = calibrate_pl.get_data(pupil_list_binocular, ref_list, mode="2d")
    # Run calibration
    print("\n\nRunning 2d binocular calibration \n\n")
    method, result = calibrate_pl.calibrate_2d_binocular(
        *matched_data_binocular, frame_size=(video_vdim, video_hdim)
    )
    # (6) Map gaze to video coordinates
    # Mapper takes two inputs: normalized pupil x and y position
    print("\n\nRunning gaze mapper [binocular] \n\n")

    # Create mapper for gaze
    if (result):
        binocular_gaze_mapper = Binocular_Gaze_Mapper(result["args"]["params"], result["args"]["params_eye0"], result["args"]["params_eye1"])
        gaze_binocular = binocular_gaze_mapper.map_batch(pupil_list_binocular)
        # Transpose output so time is the first dimension
        # TODO: Make sure the format is consistent with the monocular  gaze
        # gaze_binocular = np.vstack(gaze_binocular).T

        gaze_file = string_name.format(step="gaze")
        np.savez(gaze_file, gaze_binocular=gaze_binocular)
        final_result = True
    else:
        print("\n\nGaze Mapping Failed for Subject: ", session_id)
        final_result = False
    return final_result


def pupil_2d_monocular_v02(
    video_file_name,
    session_folder,
    sname=None,  # Base for each file?
    tag="pupil_2d_monocular_v02",
    output_path=None,
    batch_size_pupil="auto",
    progress_bar=tqdm.tqdm,
    properties=None,
):
    """
    Parameters
    ----------
    tag : A short label for the pipeline
    session_folder : string
        file path to session. Ultimately want to replace this with a session
        object from the database.
    sname : format string
        must contain {step}; if provided, tag and output_path are ignored

    Notes
    -----
    Ultimately, for saving files, we want the output of each step saved
    along with the function and parameters that were used to generate it.

    This is not the case now. Needs work.
    """
    # Deal with inputs
    if output_path is None:
        output_path = session_folder
    if sname is None:
        sname = os.path.join(output_path, "gaze_vedb", tag + "_" + video_file_name[:-4]  + "_{step}.npz")
    fdir, _ = os.path.split(sname)
    if not os.path.exists(fdir):
        print("creating", fdir)
        os.makedirs(fdir)
    # (0) Get session
    # ses = vedb_store.Session(folder=session_folder)
    # (1) Pupil detection (L, R)
    fn_pupil = pupil_detection_pl.plabs_detect_pupil

    pupil_file_left = sname.format(step="pupilpos_left")
    if os.path.exists(pupil_file_left):
        print("Loading pupils left")
        pupil_arrays_left = {}
        dat = np.load(pupil_file_left, allow_pickle=True)
        for k in dat.keys():
            pupil_arrays_left[k] = dat[k]
        pupil_list_left = gaze_utils.arraydict_to_dictlist(pupil_arrays_left)
        print(pupil_list_left[0].keys())
        print("found pupil file for: ", pupil_file_left, "\n\n")
    else:
        # Get eye files (Left eye)
        eye_left_video_file = session_folder + video_file_name
        inputs_pupil_left = dict(
            fpaths=dict(eye_video=eye_left_video_file,),
            variable_names=None,
        )
        print("\n\nRunning pupil detection for the left eye\n\n")
        # Run pupil detection
        pupil_list_left = vmp.utils.batch_run(
            fn_pupil,
            inputs_pupil_left,
            batch_size=batch_size_pupil,
            batch_combine_fn=vmp.utils.list_reduce,
            progress_bar=progress_bar,
            id=1,  # left eye # FIX ME: OPTION HERE FOR R, L, or binocular
            properties=properties,
        )
        # Get arrays instead of list of dicts
        pupil_arrays_left = gaze_utils.dictlist_to_arraydict(pupil_list_left)
        # Save pupil detection
        pupil_file_left = sname.format(step="pupilpos_left")
        np.savez(pupil_file_left, **pupil_arrays_left)
        print("\n\nSaved left pupil data into:",pupil_file_left,"\n\n")

