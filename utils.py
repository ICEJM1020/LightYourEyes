""" 
Description: 
Author: Xucheng(Timber) Zhang
Date: 2025-05-05
""" 
import numpy as np


def find_contrac_start(vel, vel_req, shift=0, judge="instant"):
    """
    Find the onset index of pupil contraction based on velocity threshold.

    Parameters:
        vel (np.ndarray): The velocity signal of the pupil size.
        vel_req (float): The threshold for contraction velocity (typically negative).
        shift (int): Number of samples over which to average for the 'period' method.
        judge (str): The judgment criterion. Either "instant" (use single point)
            or "period" (use mean over `shift` samples).

    Returns:
        int or float: Index at which contraction starts. Returns `np.nan` if no valid index is found.

    Raises:
        Exception: If `judge` is not one of "instant" or "period".
    """
    if vel_req>0: vel_req=-vel_req
    for i in range(0, len(vel)-shift):
        if judge=="instant":
            _j = vel[i]
        elif judge=="period":
            _j = np.mean(vel[i:i+shift])
        else:
            raise Exception("Judgement of Contraction need to be one of \"instant\" or \"period\"")
        if _j < vel_req:
            return i
    return np.nan


def find_stable(vel, vel_req=1, accumul_change_req=0.01, judge_window=90):
    """
    Find the index where the pupil signal stabilizes after light start.

    Parameters:
        vel (np.ndarray): The velocity signal of the pupil size.
        vel_req (float): Maximum allowed average velocity in the window.
        accumul_change_req (float): Maximum allowed cumulative change in the window.
        judge_window (int): Number of samples to use as the judging window.

    Returns:
        int or float: Index at which stability is achieved. Returns `np.nan` if not found.
    """
    for i in range(0, len(vel)-judge_window):
        if vel[i:i+judge_window].mean()<vel_req and vel[i:i+judge_window].sum()<accumul_change_req:
            return i
    return np.nan




def update_dict_key(input:dict, prefix="", surfix=""):
    _temp = {}
    for _k, _v in input.items():
        _temp[f"{prefix}{_k}{surfix}"] = _v
    return _temp



def cut_cuts(data, rounds, round_length):
    _cuts = []
    for i in range(rounds):
        _cuts.append(data[i*round_length : (i+1)*round_length])
    return _cuts


def fill_nan(arr, method="ffill"):
    """
    Fill NaN values in a 1D NumPy array using forward fill, backward fill, or both.

    Parameters:
        arr (np.ndarray): 1D numpy array with NaN values.
        method (str): 'ffill', 'bfill', or 'both'.

    Returns:
        np.ndarray: Array with NaNs filled.
    """
    if arr.ndim != 1:
        raise ValueError("Only 1D arrays are supported.")

    def _ffill(x):
        mask = np.isnan(x)
        idx = np.where(~mask, np.arange(len(x)), 0)
        np.maximum.accumulate(idx, out=idx)
        x[mask] = x[idx][mask]
        return x

    def _bfill(x):
        mask = np.isnan(x)
        idx = np.where(~mask, np.arange(len(x)), len(x) - 1)
        idx = np.minimum.accumulate(idx[::-1])[::-1]
        x[mask] = x[idx][mask]
        return x

    out = arr.copy()

    if method == "ffill":
        out = _ffill(out)

    elif method == "bfill":
        out = _bfill(out)

    elif method == "both":
        out = _ffill(out)
        out = _bfill(out)

    else:
        raise ValueError("Method must be 'ffill', 'bfill', or 'both'.")

    return out


def window_smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    adapted from SciPy Cookbook: `<https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html>`_.
    
    Parameters
    ----------    
    x: the input signal 
    window_len: the dimension of the smoothing window; should be an odd integer
    window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    Returns
    -------
    np.array: the smoothed signal        
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window should be one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='same')
    return y[(window_len-1):(-window_len+1)]


def check_welch_bins(length, cutoff, sample_rate):
    return int((length * cutoff) / sample_rate)