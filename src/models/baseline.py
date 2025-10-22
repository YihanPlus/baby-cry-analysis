# src/models/baseline.py
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import joblib


def _load_features_flat(features_path: str) -> np.ndarray:
    """Load MFCC features (.npy) shaped (N, 40, T) and flatten to (N, 40*T) for SVM."""
    X = np.load(features_path, allow_pickle=True)
    if X.ndim != 3:
        raise ValueError(f"Expected 3D MFCC array at {features_path}; got shape={X.shape}")
    return X.reshape(X.shape[0], -1)


def _read_split_csv(split_csv: str,
                    filepath_col: str = "filepath",
                    label_col: str = "label") -> tuple[list[str], np.ndarray]:
    """
    Read a split CSV with columns [filepath,label].
    Returns (filenames_in_order, labels_in_order).
    """
    df = pd.read_csv(split_csv)  # has header: filepath,label
    if filepath_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"{split_csv} must have '{filepath_col}' and '{label_col}' columns. Found {df.columns.tolist()}")
    # Keep row order as-is (assumes feature extraction used this csv order)
    fnames = df[filepath_col].apply(lambda p: os.path.basename(str(p))).tolist()
    labels = df[label_col].to_numpy()
    return fnames, labels


def _load_split_with_splitcsv(features_path: str,
                              split_csv: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load (X, y) for a split using its features .npy and the split CSV (filepath,label).
    Assumes features were generated in the same order as rows in the split CSV.
    """
    # Read labels (and implicit order) from CSV
    filenames_in_order, y = _read_split_csv(split_csv)

    # Load features and check counts
    X = _load_features_flat(features_path)
    if X.shape[0] != len(y):
        raise ValueError(
            f"Count mismatch for {features_path} vs {split_csv}: "
            f"features={X.shape[0]} labels={len(y)}.\n"
            f"Ensure the feature extraction iterated rows of the same CSV in the same order."
        )
    return X, y


def train_svm(
    train_feats="data/mfcc/train_mfcc.npy",
    val_feats="data/mfcc/val_mfcc.npy",
    test_feats="data/mfcc/test_mfcc.npy",
    train_split_csv="data/train_split.csv",
    val_split_csv="data/val_split.csv",
    test_split_csv="data/test_split.csv",
    kernel="linear",
    C=1.0,
    gamma="scale",
    model_out="models/mfcc_svm.joblib",
    tune=False,
):
    # Load aligned features + labels per split
    Xtr, ytr = _load_split_with_splitcsv(train_feats, train_split_csv)
    Xva, yva = _load_split_with_splitcsv(val_feats, val_split_csv)
    Xte, yte = _load_split_with_splitcsv(test_feats, test_split_csv)

    # Scale (fit on train only)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xva = scaler.transform(Xva)
    Xte = scaler.transform(Xte)

    # ----- OPTIONAL: hyperparameter tuning -----
    if tune:
        from sklearn.model_selection import GridSearchCV
        param_grid = {

            "kernel": ["linear", "rbf", "poly"],
            "C": [0.1, 1, 3, 10],
            "gamma": ["scale", "auto"]
        }
        base = SVC(class_weight="balanced", random_state=42)
        grid = GridSearchCV(
            estimator=base,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        grid.fit(Xtr, ytr)
        print("\n[GridSearch] best params:", grid.best_params_)
        print("[GridSearch] best CV macro-F1:", grid.best_score_)
        clf = grid.best_estimator_
    else:
        # Train SVM
        clf = SVC(kernel=kernel, C=C, gamma=gamma, class_weight="balanced", random_state=42)
        clf.fit(Xtr, ytr)

    # Evaluate
    print("\n[Validation]")
    yhat = clf.predict(Xva)
    print("Macro-F1:", f1_score(yva, yhat, average="macro"))
    print(classification_report(yva, yhat))
    print("Confusion matrix (val):\n", confusion_matrix(yva, yhat))

    print("\n[Test]")
    yhat = clf.predict(Xte)
    print("Macro-F1:", f1_score(yte, yhat, average="macro"))
    print(classification_report(yte, yhat))
    print("Confusion matrix (test):\n", confusion_matrix(yte, yhat))

    # Save model + scaler
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump({"scaler": scaler, "model": clf}, model_out)
    print(f"\n Saved SVM model to: {model_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train MFCC+SVM baseline using split CSVs.")
    ap.add_argument("--train-feats", default="data/mfcc/train_mfcc.npy")
    ap.add_argument("--val-feats", default="data/mfcc/val_mfcc.npy")
    ap.add_argument("--test-feats", default="data/mfcc/test_mfcc.npy")
    ap.add_argument("--train-split", default="data/train_split.trimmed.csv")
    ap.add_argument("--val-split", default="data/val_split.trimmed.csv")
    ap.add_argument("--test-split", default="data/test_split.trimmed.csv")
    ap.add_argument("--kernel", default="linear", choices=["linear", "rbf", "poly"])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--gamma", default="scale")
    ap.add_argument("--out", default="models/mfcc_svm.joblib")
    ap.add_argument("--tune", action="store_true", help="Run GridSearchCV for hyperparameter tuning")
    args = ap.parse_args()

    train_svm(
        train_feats=args.train_feats,
        val_feats=args.val_feats,
        test_feats=args.test_feats,
        train_split_csv=args.train_split,
        val_split_csv=args.val_split,
        test_split_csv=args.test_split,
        kernel=args.kernel,
        C=args.C,
        gamma=args.gamma,
        model_out=args.out,
        tune=args.tune
    )
