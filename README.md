Copyright © Westcliff

This is a school project.

# 👶 Baby Cry Analysis — concise & practical

Small, reproducible pipeline to turn baby-cry WAVs into MFCC features and train classifiers (baseline SVM and a compact CNN).

## Quickstart

1. Use Python 3.10–3.13 (the pinned requirements currently exclude 3.14). Create venv and install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. Run the notebooks in order: `notebooks/01_explore_data.ipynb`, `02_feature_extraction.ipynb`, then `03_baseline_model.ipynb`.
3. Or run the CNN script directly:

```bash
python src/models/cnn.py
```

## Data preprocessing

- Discover and label `.wav` files (from filenames) → `data/audio_filepaths.csv`.
- Stratified splits: train / val / test (CSV files in `data/`).
- Normalize waveforms: compute mean/std on the _training_ set and apply to all splits (saved to `data/normalized_wav/`).
- Trim/pad clips to fixed duration (8 s) and resample (SR used in notebooks is 16 kHz).
- Extract MFCCs: 40 coefficients per frame; resulting arrays saved in `data/mfcc/`.
- Make CNN inputs: pad/trim time axis to a fixed number of frames (typical shape used is `(N, 40, 173, 1)`) and save to `data/mfcc_cnn/`.
- Label binning: continuous labels are converted to discrete classes using percentile cut points (saved as `*_labels_binned.npy`).

## Model (CNN)

- Architecture: 3 × Conv2D blocks (32 → 64 → 128 filters) each with BatchNorm, ReLU, MaxPool, Dropout; final AdaptiveAvgPool → FC.
- Input shape: (batch, 1, time_frames, n_mfcc) — built from `data/mfcc_cnn/*`.
- Training: Adam optimizer (lr=1e-3), CrossEntropyLoss, default run uses ~50 epochs (script-configurable). Batch size typically 32.
- Outputs: saved model weights `models/baby_cry_cnn_mel.pt` and training plot `models/train_val_plot.png`.

## Important paths

- Raw audio: `data/raw/`
- Normalized audio: `data/normalized_wav/{train,val,test}`
- MFCC arrays: `data/mfcc/*.npy`
- CNN tensors & binned labels: `data/mfcc_cnn/*_mfcc_cnn.npy`, `*_labels_binned.npy`
- Models & plots: `models/`

## Quick checks & tips

- If pip complains about Python 3.14, create a pyenv/conda env with Python 3.13.
- Confirm array shapes before training:

```python
import numpy as np
arr = np.load('data/mfcc_cnn/train_mfcc_cnn.npy', allow_pickle=True)
print('shape:', arr.shape)
```

---
