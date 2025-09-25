# src/preprocessing.py
"""
Audio preprocessing utilities:
 - resample to target sample rate
 - trim or pad to fixed duration
 - normalize amplitude
"""

import librosa
import numpy as np


# Global defaults
TARGET_SR = 16000        # resample everything to 16 kHz (best for cries)
FIXED_DURATION = 8.0     # target length in seconds (based on dataset)


def load_and_standardize(path, sr=TARGET_SR, duration=FIXED_DURATION):
    """
    Load an audio file, resample, trim/pad to fixed duration, normalize.

    Returns:
        y (np.ndarray): waveform (1D)
        sr (int): sampling rate
    """
    # Load and resample
    y, _sr = librosa.load(path, sr=sr, mono=True)

    # Ensure fixed length
    target_len = int(sr * duration)
    if len(y) > target_len:
        start = (len(y) - target_len) 
        y = y[start:start + target_len] # center crop
    else:
        y = np.pad(y, (0, target_len - len(y))) # zero pad

    # Normalize amplitude
    peak = np.max(np.abs(y)) or 1.0
    y = y / peak

    return y, sr
