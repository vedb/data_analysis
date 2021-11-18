import yaml
import numpy as np


def read_yaml(parameters_fpath):
    """ Read configuration yaml file

    Parameters
    ----------
    parameters_fpath: str
        path to the yaml file

    Returns
    ----------
    paraam_dict: dict
        Returns the yaml parameters as a Python Dictionary
    """
    param_dict = dict()
    with open(parameters_fpath, "r") as stream:
        param_dict = yaml.safe_load(stream)
    return param_dict


def find_closest_timestamp_index(ts, timestamp_array):
    """ Find the closest index to the ts value among the timestamp_array

    Parameters
    ----------
    timestamp_array: list or numpy array
        An array of timestamp values

    ts: float, len 1
        Desired timestamp value

    Returns
    ----------
    index: int
        Returns the index value of the closest matching timestamp
    """
    return np.argmin(np.abs((timestamp_array - ts).astype(float)))
