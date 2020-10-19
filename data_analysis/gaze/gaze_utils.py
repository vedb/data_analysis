# Utilities supporting gaze calibration

import numpy as np
import cv2


def onoff_from_binary(data, return_duration=True):
    """Converts a binary variable data into onsets, offsets, and optionally durations
    
    This may yield unexpected behavior if the first value of `data` is true. 
    
    Parameters
    ----------
    data : array-like, 1D
        binary array from which onsets and offsets should be extracted
    
    """
    data = data.astype(np.float).copy()
    ddata = np.hstack([[0], np.diff(data)])
    onsets, = np.nonzero(ddata > 0)
    #print(onsets)
    offsets, = np.nonzero(ddata < 0)
    #print(offsets)
    onset_first = onsets[0] < offsets[0]
    len(onsets) == len(offsets)

    on_at_end = False
    on_at_start = False
    if onset_first:
        if len(onsets) > len(offsets):
            offsets = np.hstack([offsets, [-1]])
            on_at_end = True
    else:
        if len(offsets) > len(onsets):
            onsets = np.hstack([-1, offsets])
            on_at_start = True
    onoff = np.vstack([onsets, offsets])
    if return_duration:
        duration = offsets - onsets
        if on_at_end:
            duration[-1] = len(data) - onsets[-1]
        if on_at_start:
            duration[0] = offsets[0] - 0
        onoff = np.vstack([onoff, duration])
    
    onoff = onoff.T.astype(np.int)
    return onoff


def time_to_index(onsets_offsets, timeline):
    """find indices between onsets & offsets in timeline

    Parameters
    ----------
    """
    out = np.zeros_like(onsets_offsets)
    for ct, (on, off) in enumerate(onsets_offsets):
        i = np.flatnonzero(timeline > on)[0]
        j = np.flatnonzero(timeline < off)[-1]
        out[ct] = [i, j]
    return out


def dictlist_to_arraydict(dictlist):
    """Convert from pupil format list of dicts to dict of arrays"""
    dict_fields = list(dictlist[0].keys())
    out = {}
    for df in dict_fields:
        out[df] = np.array([d[df] for d in dictlist])
    return out


def arraydict_to_dictlist(arraydict):
    """Convert from dict of arrays to pupil format list of dicts"""
    dict_fields = list(arraydict.keys())
    first_key = dict_fields[0]
    n = len(arraydict[first_key])
    out = []
    for j in range(n):
        frame_dict = {}
        for k in dict_fields:
            value = arraydict[k][j]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            frame_dict[k] = value
        out.append(frame_dict)
    return out
