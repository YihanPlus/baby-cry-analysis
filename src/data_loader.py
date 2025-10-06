import os
import glob
import pandas as pd
from scripts.parse_labels import parse_labels 

def get_audio_file_paths(data_dir="data/raw", pattern="*.wav"):
    """
    Recursively collect all audio file paths from a directory (including subfolders).

    Returns:
        files (list of str): list of full file paths.
        labels (list of str): corresponding labels extracted from filenames.
    """
    files = glob.glob(os.path.join(data_dir, "**", pattern), recursive=True)
    files.sort()
    labels = parse_labels(files)
    return files, labels

def export_file_list_to_csv(files, labels, output_path="data/audio_filepaths.csv"):
    """
    Save file paths and labels to a CSV for reference, without overwriting
    the existing labels_from_filenames.csv metadata file.
    """
    df = pd.DataFrame({"filepath": files, "label": labels})
    df.to_csv(output_path, index=False)
    print(f"Exported file list to {output_path}")
