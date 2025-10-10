# 👶 Baby Cry Analysis

This project classifies infant cries into emotional or physiological states using audio-based features and machine learning.

## 📘 Project Overview
- Input: raw `.wav` recordings of baby cries  
- Output: predicted cry categories such as hunger, tiredness, discomfort, etc.  
- Models: baseline SVM (MFCC features) and CNN (spectrogram inputs)

## 🧩 Data Preprocessing Pipeline
The preprocessing ensures all audio clips are consistent in duration, sampling rate, and amplitude before feature extraction.

**Steps:**
1. Collect and label all `.wav` files under `data/raw/`
2. Split into train (68%), validation (12%), test (20%)
3. Normalize waveform amplitude and trim silence
4. Extract 40 MFCCs per sample (8 s duration)
5. Convert to fixed-size tensors for CNN: `(N, 40, 173, 1)`

**Outputs:**
- `data/mfcc/*.npy` → SVM inputs  
- `data/mfcc_cnn/*.npy` → CNN inputs