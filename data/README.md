# 📂 `data/` Folder Overview

This folder contains all datasets and intermediate outputs used in the **Baby Cry Analysis** project.

### **Contents**
- `raw/` – original baby cry `.wav` recordings (source data, unmodified)  
- `normalized_wav/` – audio after resampling, silence trimming, and amplitude normalization  
- `mfcc/` – extracted MFCC features (used for the SVM baseline model)  
- `mfcc_cnn/` – fixed-size MFCC tensors `(N, 40, 173, 1)` prepared for CNN input  
- `audio_filepaths.csv` – full list of audio file paths and inferred labels  
- `train_split.csv`, `val_split.csv`, `test_split.csv` – reproducible dataset splits  

---

### **Data Workflow**
| Step | Output |
|------|---------|
| raw audio | → `audio_filepaths.csv` (via `parse_labels.py`) |
| split train/val/test | → `normalized_wav/` |
| extract MFCCs | → `mfcc/` |
| prepare CNN tensors | → `mfcc_cnn/` |

---
### **Notes**
- `raw/` should remain unchanged (source data only).  
- All other folders can be safely regenerated using notebooks or scripts under `src/`.  
- Large `.wav` files should be ignored by Git (`.gitignore`).