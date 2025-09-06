import numpy as np
import pandas as pd



def calculate_cv(arr: np.ndarray) -> float:
    """
    Computes the coefficient of variation (CV) for a 1D NumPy array.

    Args:
        arr: A 1D NumPy array of numbers.

    Returns:
        The coefficient of variation as a float. Returns np.nan if the mean is zero.
    """
    if not isinstance(arr, np.ndarray) or arr.ndim != 1:
        raise ValueError("Input must be a 1D NumPy array.")

    mean = np.mean(arr)
    std_dev = np.std(arr)

    if mean == 0:
        return np.nan  # CV is undefined if the mean is zero.
    else:
        return std_dev / mean


import numpy as np

def calculate_velocity(pupil_data: np.ndarray, sample_rate: float) -> np.ndarray:
    """
    Calculates the velocity of pupil diameter change.
    
    Args:
        pupil_data (np.ndarray): 1D array of pupil diameter values in mm.
        sample_rate (float): Sample rate in Hz (samples per second).
    
    Returns:
        np.ndarray: Array of velocity values in mm/s. Same length as input.
    """
    # Calculate differences between consecutive samples
    diameter_diff = np.diff(pupil_data)
    time_diff = 1.0 / sample_rate  # Time between samples in seconds
    
    # Calculate velocity
    velocity = diameter_diff / time_diff
    
    # Pad with 0 at the beginning to maintain same length as input
    velocity = np.concatenate([[0], velocity])
    
    return np.abs(velocity)

def calculate_latency(pupil_data: np.ndarray, velocity: np.ndarray, sample_rate: float, 
                     stimulus_onset_frame: int, std_threshold: int = 3) -> tuple[float, float]:
    """
    Calculates the latency of constriction.
    
    Args:
        pupil_data (np.ndarray): 1D array of pupil diameter values in mm.
        velocity (np.ndarray): 1D array of velocity values in mm/s.
        sample_rate (float): Sample rate in Hz.
        stimulus_onset_frame (int): Frame index when the stimulus occurred.
        std_threshold (int): Number of standard deviations to use for threshold.
    
    Returns:
        tuple[float, float]: Latency in ms and absolute time of constriction onset in seconds.
    """
    # Calculate baseline statistics from pre-stimulus data
    if stimulus_onset_frame <= 0:
        return np.nan, np.nan
    
    baseline_velocity = velocity[stimulus_onset_frame:]
    baseline_velocity_mean = np.mean(baseline_velocity)
    baseline_velocity_std = np.std(baseline_velocity)
    
    # Calculate threshold for constriction detection
    latency_threshold = baseline_velocity_mean + std_threshold * baseline_velocity_std
    
    threshold_crossings = np.where(baseline_velocity > latency_threshold)[0]
    
    if len(threshold_crossings) == 0:
        return np.nan, np.nan
    
    # Find first threshold crossing
    onset_frame = stimulus_onset_frame + threshold_crossings[0]
    
    # Convert to time
    stimulus_onset_time = stimulus_onset_frame / sample_rate
    latency_onset_time = onset_frame / sample_rate
    latency_ms = (latency_onset_time - stimulus_onset_time) * 1000
    
    return latency_ms, latency_onset_time

def calculate_average_transition_speed(velocity: np.ndarray, sample_rate: float,
                                     latency_onset_time: float, peak_constriction_frame: float) -> float:
    """
    Calculates the average speed during the constriction transition.
    
    Args:
        velocity (np.ndarray): 1D array of velocity values in mm/s.
        sample_rate (float): Sample rate in Hz.
        latency_onset_time (float): Absolute time when constriction started (seconds).
        peak_constriction_time (float): Absolute time of peak constriction (seconds).
    
    Returns:
        float: Average velocity in mm/s during the transition.
    """
    # Convert times to frame indices
    latency_onset_frame = int(latency_onset_time * sample_rate)
    
    # Ensure valid frame range
    latency_onset_frame = max(0, latency_onset_frame)
    peak_constriction_frame = min(len(velocity) - 1, peak_constriction_frame)
    
    if latency_onset_frame >= peak_constriction_frame:
        return np.nan
    
    # Calculate average velocity during transition
    transition_velocity = velocity[latency_onset_frame:peak_constriction_frame + 1]
    average_speed = np.mean(transition_velocity)
    
    return average_speed

def analyze_pupil_response(pupil_data: np.ndarray, sample_rate: float, 
                          stimulus_onset_frame: int, color_terminal_frame: int, std_threshold: int = 3,) -> dict:
    """
    Complete analysis of pupil light reflex response.
    
    Args:
        pupil_data (np.ndarray): 1D array of pupil diameter values in mm.
        sample_rate (float): Sample rate in Hz.
        stimulus_onset_frame (int): Frame index when the stimulus occurred.
        std_threshold (int): Number of standard deviations for latency threshold.
    
    Returns:
        dict: Dictionary containing all calculated metrics.
    """
    # Calculate velocity
    velocity = calculate_velocity(pupil_data, sample_rate)
    
    # Calculate latency
    latency_ms, latency_onset_time = calculate_latency(
        pupil_data, velocity, sample_rate, stimulus_onset_frame, std_threshold
    )
    
    # Calculate average transition speed
    if not np.isnan(latency_onset_time):
        avg_transition_speed = calculate_average_transition_speed(
            velocity, sample_rate, latency_onset_time, color_terminal_frame
        )
    else:
        avg_transition_speed = calculate_average_transition_speed(
            velocity, sample_rate, stimulus_onset_frame / sample_rate, color_terminal_frame
        )
    
    return {
        'latency_ms': latency_ms,
        'latency_onset_time': latency_onset_time - stimulus_onset_frame/sample_rate,
        'average_transition_speed': avg_transition_speed,
    }