import pupil_detectors
import numpy as np



def plabs_detect_pupil(eye_video, timestamps=None, progress_bar=None, id=None, properties=None, **kwargs):
    """

    This is a simple wrapper to allow Pupil Labs `pupil_detectors` code
    to process a whole video of eye data. 

    Parameters
    ----------
    eye_video : array
        video to parse,  (t, x, y, [color]), already loaded as uint8 data. 
    timestamps : array
        optional, time associated with each frame in video. If provided,
        timestamps for each detected pupil are returned with 
    progress_bar : tqdm object or None
        if tqdm object is provided, a progress bar is displayed 
        as the video is processed.
    id : int, 0 or 1
        ID for eye (eye0, i.e. left, or eye1, i.e. right)
    
    Notes
    -----
    Parameters for Pupil Detector2D object, passed as a dict called
    `properties` to `pupil_detectors.Detector2D()`; fields are:
        coarse_detection = True
        coarse_filter_min = 128
        coarse_filter_max = 280
        intensity_range = 23
        blur_size = 5
        canny_treshold = 160
        canny_ration = 2
        canny_aperture = 5
        pupil_size_max = 100
        pupil_size_min = 10
        strong_perimeter_ratio_range_min = 0.6
        strong_perimeter_ratio_range_max = 1.1
        strong_area_ratio_range_min = 0.8
        strong_area_ratio_range_max = 1.1
        contour_size_min = 5
        ellipse_roundness_ratio = 0.09    # HM! Try setting this?
        initial_ellipse_fit_treshhold = 4.3
        final_perimeter_ratio_range_min = 0.5
        final_perimeter_ratio_range_max = 1.0
        ellipse_true_support_min_dist = 3.0
        support_pixel_ratio_exponent = 2.0    
    Returns
    -------
    pupil_dicts : list of dicts
        dictionary for each detected instance of a pupil. Each 
        entry has fields: 
        luminance
        timestamp [if timestamps are provided, which they should be]
        norm_pos
    """
    if progress_bar is None:
        progress_bar = lambda x : x
    # Specify detection method later?
    if properties is None:
        properties = {}
    det = pupil_detectors.Detector2D(properties=properties)
    n_frames, eye_vdim, eye_hdim = eye_video.shape[:3]
    eye_dims = np.array([eye_hdim, eye_vdim])
    pupil_dicts = []
    lum = np.zeros((n_frames,))
    for frame in progress_bar(range(n_frames)):
        fr = eye_video[frame].copy()
        # If an extra dimension is present, assume eye video is spuriously RGB
        if np.ndim(fr) == 3:
            fr = fr[:,:,0]
        # Pupil needs c-ordered arrays, so switch from default load:
        fr = np.ascontiguousarray(fr)
        # Call detector & process output
        out = det.detect(fr)
        # Get rid of raw data as input; no need to keep
        if 'internal_2d_raw_data' in out:
            _ = out.pop('internal_2d_raw_data')
        # Save average luminance of eye video for reference
        out['luminance'] = fr.mean()
        # Normalized position
        out['norm_pos'] = (np.array(out['location']) / eye_dims).tolist()
        if timestamps is not None:
            out['timestamp'] = timestamps[frame]
        if id is not None:
            out['id'] = id
        pupil_dicts.append(out)
    return pupil_dicts

