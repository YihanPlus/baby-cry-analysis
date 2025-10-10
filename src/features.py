"""
Feature extraction utilities:
 - extract_mfcc(): extract MFCCs from a single audio file
 - extract_split_mfcc(): batch extract MFCCs for a dataset split
 - pad_or_trim_mfcc(): pad/truncate MFCC to fixed time frames
 - prepare_cnn_input(): convert MFCC list to 4D CNN-ready tensor
"""

import os
import numpy as np
import librosa
from tqdm import tqdm

def extract_mfcc(file_path, n_mfcc=40, sr=16000, duration=8.0):
    """
    Extract MFCCs from one normalized audio file.
    Returns MFCC array of shape [n_mfcc, time_frames].
    """
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc


def extract_split_mfcc(split_name, folder_path, out_dir="data/mfcc", n_mfcc=40, sr=16000, duration=8.0):
    """
    Extract MFCC features for all .wav files in a split (train/val/test).
    Saves results to data/mfcc/{split_name}_mfcc.npy
    """
    os.makedirs(out_dir, exist_ok=True)
    filepaths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".wav")]

    X = []
    for f in tqdm(filepaths, desc=f"Extracting MFCCs ({split_name})"):
        mfcc = extract_mfcc(f, n_mfcc=n_mfcc, sr=sr, duration=duration)
        X.append(mfcc)

    X = np.array(X, dtype=object)
    np.save(f"{out_dir}/{split_name}_mfcc.npy", X)
    print(f"✅ Saved {split_name} MFCCs to {out_dir}/{split_name}_mfcc.npy")
    return X

def pad_or_trim_mfcc(mfcc, max_frames=173):
    """Pad or truncate MFCC matrix to fixed number of frames for CNN."""
    if mfcc.shape[1] < max_frames:
        pad_width = max_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_frames]
    return mfcc


def prepare_cnn_input(mfcc_list, max_frames=173):
    """Convert a list of MFCCs into a 4D CNN-ready tensor."""
    X = np.stack([pad_or_trim_mfcc(m, max_frames=max_frames) for m in mfcc_list]).astype(np.float32)
    X = X[..., np.newaxis]  # add channel dimension
    return X