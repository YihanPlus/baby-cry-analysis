"""
Audio preprocessing utilities:
 - load_and_standardize(): resample, trim/pad, peak normalize
 - fit_audio_normalizer(): compute dataset-level mean/std
 - apply_audio_normalization(): apply z-score normalization
"""

import numpy as np
import librosa

def load_and_standardize(path, sr=16000, duration=8.0):
    """
    Load an audio file, resample, trim/pad to fixed duration, peak normalize.
    Used mainly for data exploration or visualization.
    """
    y, _sr = librosa.load(path, sr=sr, mono=True)
    target_len = int(sr * duration)
    if len(y) > target_len:
        start = (len(y) - target_len)
        y = y[start:start + target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))
    peak = np.max(np.abs(y)) or 1.0
    y = y / peak
    return y, sr


# --- new functions for training normalization ---
def fit_audio_normalizer(train_files, sr=16000):
    """
    Compute global mean and std of waveform amplitudes from training set.
    This should only be run on training files to avoid data leakage.
    """
    all_samples = []
    for f in train_files:
        y, _ = librosa.load(f, sr=sr, mono=True)
        all_samples.append(y)
    concat = np.concatenate(all_samples)
    mean = np.mean(concat)
    std = np.std(concat)
    return mean, std


def apply_audio_normalization(file_path, mean, std, sr=16000):
    """
    Apply dataset-level normalization (z-score) to a single file.
    """
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    y_norm = (y - mean) / std
    return y_norm, sr
