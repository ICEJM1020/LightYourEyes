""" 
Description: 
Author: Xucheng(Timber) Zhang
Date: 2025-04-17
""" 

import numpy as np
from scipy.signal import butter, filtfilt


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


def find_blink_intervals_v0(raw_data, sample_rate, bs_abnormal_tol=1.0, be_abnormal_tol=0.5, blink_gap=9, max_blink_dura=45, smooth_win=20):
    """
    Detects blink intervals in pupil data by identifying sharp changes in signal derivative.

    Parameters:
        raw_data (np.ndarray): The raw pupil diameter signal.
        sample_rate (int): Sampling rate in Hz.
        bs_abnormal_tol (float): Threshold multiplier for detecting sharp negative slope (blink start).
        be_abnormal_tol (float): Threshold multiplier for detecting sharp positive slope (blink end).
        blink_gap (int): Maximum gap (in samples) between adjacent blink events to consider merging.
        max_blink_dura (int): Maximum allowed blink duration in samples.

    Returns:
        List[Tuple[int, int]]: A list of tuples representing (start, end) indices of blink intervals.

    Notes:
        - Uses derivative thresholding to detect blink-like drops and rises in the signal.
        - Merges close blink intervals based on `blink_gap`.
        - Requires a `window_smooth` function defined elsewhere to reduce noise.
    """
    smooth_raw = window_smooth(raw_data, smooth_win)

    deriv = np.gradient(smooth_raw) * sample_rate
    deriv_std = np.std(deriv)

    # Threshold to detect anomaly in derivative
    neg_anomaly = np.where(deriv < (-bs_abnormal_tol * deriv_std))[0].tolist()
    posi_anomaly = np.where(deriv > be_abnormal_tol * deriv_std)[0].tolist()
    if not neg_anomaly or not posi_anomaly:
        return []
    
    _temp_neg = [neg_anomaly[0]]
    _temp_posi = [posi_anomaly[0]]
    anomaly_intervals = []
    i, j = 1,1
    while 1:
        while i<len(neg_anomaly) and neg_anomaly[i] - _temp_neg[-1]==1:
            _temp_neg.append(neg_anomaly[i])
            i += 1
        
        while j<len(posi_anomaly) and posi_anomaly[j] - _temp_posi[-1]==1:
            _temp_posi.append(posi_anomaly[j])
            j += 1

        if  0 <= _temp_posi[0] - _temp_neg[-1] <= max_blink_dura:
            l = max(0, min(_temp_neg))
            r = min(len(raw_data), max(_temp_posi))
            anomaly_intervals.append((l , r))
            if i+1<len(neg_anomaly) and j+1<len(posi_anomaly):
                _temp_neg = [neg_anomaly[i]]
                _temp_posi = [posi_anomaly[j]]
                i+=1
                j+=1
            else:
                break
        elif _temp_posi[-1] < _temp_neg[0]:
            if j < len(posi_anomaly):
                _temp_posi = [posi_anomaly[j]]
                j+=1
            else:
                break
        else:
            if (i < len(neg_anomaly)) and (neg_anomaly[i]<_temp_posi[0]):
                _temp_neg.append(neg_anomaly[i])
                i += 1
            elif (i < len(neg_anomaly)) and (neg_anomaly[i] > _temp_posi[-1]):
                _temp_neg = [neg_anomaly[i]]
                i += 1
            else:
                break
    if len(anomaly_intervals)==0: return []
            
    merged = [anomaly_intervals[0]]

    for start, end in anomaly_intervals[1:]:
        last_start, last_end = merged[-1]

        if start - last_end <= blink_gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


# 使用迭代方法的版本（更易理解）
def find_blink_intervals(raw_data: np.ndarray, blink_gap=10):
    """
    使用迭代方法的版本，逻辑更清晰
    """
    if raw_data.ndim != 1:
        raise ValueError("输入数据必须是一维数组")
    
    isnan_mask = np.isnan(raw_data)
    intervals = []
    i = 0
    n = len(raw_data)
    
    while i < n:
        # 找到NaN段的开始
        if isnan_mask[i]:
            start = i
            # 找到NaN段的结束
            while i < n and isnan_mask[i]:
                i += 1
            end = i - 1
            intervals.append([start, end])
        else:
            i += 1

    if len(intervals)==0:
        return []

    merged = [intervals[0]]

    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]

        if start - last_end <= blink_gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    
    return merged


def deBlink(raw_data, flatten_shift=2, max_blink_dura=45):
    """
    Remove blink-induced anomalies from raw time-series data by flattening regions
    of rapid change (i.e., blinks) using a linear interpolation strategy.

    This function computes the derivative of the input data to detect sudden large
    fluctuations indicative of blink artifacts. Detected anomalous segments are
    then smoothed by replacing the region with a linear interpolation between
    surrounding data points, effectively "flattening" the blink region.

    Parameters:
        raw_data (np.ndarray): The original pupil diameter data.
        flatten_shift (int): Buffer (in samples) to extend before and after each blink for smoothing.
        blink_start_vel_tol (float): Velocity threshold for detecting blink onset.
        blink_end_vel_tol (float): Velocity threshold for detecting blink offset.
        max_blink_dura (int): Maximum allowed blink duration in samples.
        sample_rate (int): Sampling rate in Hz.

    Returns:
        np.ndarray: Blink-corrected signal with flattened blink segments.

    Example:
        >>> clean_signal = deBlink(raw_signal, flatten_shift=5)
        >>> plt.plot(raw_signal, label="Raw")
        >>> plt.plot(clean_signal, label="Deblinked")
        >>> plt.legend()
        >>> plt.show()

    Notes:
        - Blink intervals are smoothed with linear interpolation and small Gaussian noise.
        - The goal is to make blink periods look like smooth transitions instead of sharp drops.
    """
    data_flat = raw_data.copy()

    intervals = find_blink_intervals(
        raw_data=raw_data,
        blink_gap=max_blink_dura
    )

    half_flatten_shift = int(flatten_shift//2)
    for l, r in intervals:
        start = max(0, l - flatten_shift - half_flatten_shift)
        end = min(len(raw_data)-1, r + flatten_shift+half_flatten_shift)

        noise = np.random.normal(loc=0.0, scale=0.05, size=end-start)
        data_flat[start:end] = np.linspace(
                np.nanmean(raw_data[start:start+half_flatten_shift+1]), 
                np.nanmean(raw_data[end-half_flatten_shift:end]), 
                end-start
            ) + noise

    return data_flat


def lowPass(raw_data, cutoff, fs, order=4):
    """
    Apply a low-pass Butterworth filter to the input signal.

    This function smooths the input time-series data by attenuating frequencies
    higher than the specified cutoff. A zero-phase filtering approach is used
    (`filtfilt`) to prevent phase distortion in the filtered output.

    Parameters:
        raw_data (np.ndarray): 1D array of input signal data to be filtered.
        cutoff (float): Cutoff frequency of the low-pass filter in Hz.
        fs (float): Sampling frequency of the input signal in Hz.
        order (int): Order of the Butterworth filter (default is 4).

    Returns:
        np.ndarray: The filtered signal as a 1D NumPy array, with the same shape
        as `raw_data`.

    Raises:
        ValueError: If the normalized cutoff frequency is not between 0 and 1.

    Example:
        >>> filtered = lowpass_filter(raw_data, cutoff=4.0, fs=60.0)
        >>> plt.plot(raw_data, label='Raw')
        >>> plt.plot(filtered, label='Filtered')
        >>> plt.legend()
        >>> plt.show()

    Notes:
        - The Nyquist frequency is half the sampling frequency.
        - `filtfilt` applies the filter forward and backward to eliminate phase shift.
        - The filter is ideal for removing noise from physiological data.
    """
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    
    if not 0 < normal_cutoff < 1:
        raise ValueError(f"Normalized cutoff frequency must be between 0 and 1. Got {normal_cutoff}")

    # Get filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Apply filter (zero-phase)
    filtered_data = filtfilt(b, a, raw_data)
    return filtered_data



def transPercent(data, method='mean', base_range=None, fs=None):
    """
    Convert time-series data to percent change relative to a computed baseline.

    This function calculates the percent change of a signal based on a specified
    baseline computation method. It supports using the entire dataset, a fixed
    time range, or multiple time ranges to compute the baseline.

    Parameters:
        data (array-like): Input time-series data to transform.
        method (str): Baseline computation method. Options are:
            - 'mean': Use the mean of the entire data as the baseline.
            - 'median': Use the median of the entire data as the baseline.
            - 'range_mean': Use the mean of one or more time intervals (in seconds)
              as the baseline. Requires `base_range` and `fs`.
        base_range (Tuple[float, float] or List[Tuple[float, float]], optional):
            A single (start, end) time tuple or a list of such tuples (in seconds),
            defining the interval(s) over which to compute the baseline. Required
            if `method` is 'range_mean'.
        fs (float, optional): Sampling frequency of the data in Hz. Required
            if `method` is 'range_mean'.

    Returns:
        np.ndarray: A NumPy array representing the percent change of the input
        data relative to the computed baseline, calculated as:
        ((data - baseline) / baseline) * 100

    Raises:
        ValueError: If `method` is 'range_mean' but `base_range` or `fs` is not provided.
        ValueError: If `method` is not one of 'mean', 'median', or 'range_mean'.

    Example:
        >>> transPercent(data, method='mean')
        >>> transPercent(data, method='range_mean', base_range=(0, 1), fs=100)
        >>> transPercent(data, method='range_mean', base_range=[(0, 1), (2, 3)], fs=100)

    Notes:
        - Multiple intervals in `base_range` are averaged together to form the final baseline.
        - This is commonly used in physiological and behavioral signal analysis
          to normalize responses or highlight deviations from a defined baseline.
    """
    data = np.asarray(data)

    if method == 'mean':
        base = np.mean(data)
    elif method == 'median':
        base = np.median(data)
    elif method == 'max':
        base = np.max(data)
    elif method == 'range_mean':
        if base_range is None or fs is None:
            raise ValueError("For 'range_mean', you must provide base_range and fs.")
        
        if type(base_range[0])==list:
            _base = []
            for _s, _e in base_range:
                start_idx = int(_s * fs)
                end_idx = int(_e * fs)
                _base.append(np.mean(data[start_idx:end_idx]))
            base = np.mean(_base)
        else:
            start_idx = int(base_range[0] * fs)
            end_idx = int(base_range[1] * fs)
            base = np.mean(data[start_idx:end_idx])
    else:
        raise ValueError("Invalid method. Choose from 'mean', 'median', or 'range_mean'.")

    return ((data - base) / base ) * 100


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


