#!/usr/bin/env python3
"""
scripts/parse_labels.py

Parse labels from filenames and write a CSV:
  data/labels_from_filenames.csv

Usage:
  python3 scripts/parse_labels.py \
  --root data/raw/baby_cry_sence_dataset \
  --out data/labels_from_filenames.csv

  python3 scripts/parse_labels.py --root data/raw/baby_cry_sence_dataset --out data/labels_from_filenames.csv --compute-duration

Notes:
 - Recognizes short codes in filenames: bp, bu, ch, dc, hu, lo, sc, ti
 - If compute-duration is enabled, librosa is used (optional, slower)
"""
import os
import re
import argparse
from collections import Counter
import csv

# mapping short code -> canonical label
SHORT_MAP = {
    "bp": "belly pain",
    "bu": "burping",
    "ch": "cold hot",
    "dc": "discomfort",
    "hu": "hungry",
    "lo": "lonely",
    "sc": "scared",
    "ti": "tired",
}

AUDIO_EXTS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')

def infer_label_from_filename(fname):
    """
    Try to find any of the short codes in filename tokens.
    Returns (short_code or 'unknown', canonical_label or 'unknown')
    """
    base = os.path.basename(fname).lower()
    # split on non-alphanumeric to get tokens like '69bda5d6', '1436936185', '1', '1', 'm', '26', 'bp'
    tokens = re.split(r'[^0-9a-z]+', base)
    tokens = [t for t in tokens if t]
    # first pass: token exact match (fast & robust)
    for t in tokens[::-1]:  # check from end first (codes often near filename end)
        if t in SHORT_MAP:
            return t, SHORT_MAP[t]
    # second pass: substring match (fallback)
    for code in SHORT_MAP:
        if re.search(r'[^0-9a-z]{}(?:[^0-9a-z]|$)'.format(re.escape(code)), base):
            return code, SHORT_MAP[code]
    # final fallback: unknown
    return 'unknown', 'unknown'

def find_audio_files(root):
    files = []
    for dirpath, _, fnames in os.walk(root):
        for f in fnames:
            if f.lower().endswith(AUDIO_EXTS):
                files.append(os.path.join(dirpath, f))
    return sorted(files)

def main(root, out_csv, compute_duration=False):
    files = find_audio_files(root)
    rows = []
    counts = Counter()
    unknowns = []
    if compute_duration:
        try:
            import librosa
        except Exception as e:
            raise RuntimeError("To compute duration, librosa must be installed. Install it or run without --compute-duration.") from e

    for p in files:
        short, label = infer_label_from_filename(p)
        rec = {"filename": p, "label_short": short, "label": label}
        if compute_duration:
            try:
                y, sr = librosa.load(p, sr=None, mono=True)
                duration = float(len(y)) / float(sr)
            except Exception:
                duration = None
            rec["duration_sec"] = duration
        rows.append(rec)
        counts[label] += 1
        if label == "unknown":
            unknowns.append(p)

    # write CSV
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fieldnames = ["filename", "label_short", "label"] + (["duration_sec"] if compute_duration else [])
    with open(out_csv, "w", newline='', encoding="utf8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    total = sum(counts.values())
    print(f"Found {len(files)} audio files under {root}. Wrote labels to {out_csv}")
    print("Class counts:")
    for k, v in counts.most_common():
        pct = v / total * 100 if total>0 else 0
        print(f"  {k:12s} : {v:4d}  ({pct:5.2f}%)")
    if unknowns:
        print("\nWarning: unknown labels detected for these files (showing up to 10):")
        for u in unknowns[:10]:
            print("  ", u)
    print("\nDone.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/raw", help="root folder containing audio files")
    p.add_argument("--out", default="data/labels_from_filenames.csv", help="output CSV path")
    p.add_argument("--compute-duration", action="store_true", help="compute audio durations (requires librosa, slower)")
    args = p.parse_args()
    main(args.root, args.out, args.compute_duration)
