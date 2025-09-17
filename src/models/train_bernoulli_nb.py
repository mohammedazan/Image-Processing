#!/usr/bin/env python3
"""
train_bernoulli_nb.py

README (usage and assumptions)
------------------------------
Baseline Bernoulli Naive Bayes training for Tree Species Classification.
Accepts Hu 2D features, PCA 3D features, or both. Converts numeric features
into binary via thresholding learned ONLY on the training set.

Inputs (defaults can be changed via CLI):
- --hu_csv: data/processed/2d_hu/hu_features_table.csv
- --pca_csv: data/processed/descriptors/descriptors_table_pca.csv
- --labels_csv: CSV with columns id,label (e.g., data/splits/train.csv)
- --test_csv: optional CSV with column id for final evaluation only

Key functionality:
- Align features with labels by id
- Handle NaNs: drop all-NaN cols/rows, impute remaining NaNs with training-set median
- Optional z-score before binarization
- Binarization methods: median (default), quantile, otsu
- Thresholds are saved to YAML under experiments/bernoulli/bin_thresholds.yaml (or debug path)
- Stratified holdout or stratified k-fold CV
- Optional alpha grid search for BernoulliNB using macro F1
- Save model, metrics, confusion matrices, config, split indices, and summary

Assumptions:
- Labels CSV has columns: id,label
- Feature tables have at least columns: id, saved_path or per-feature columns; we expect
  a row per id and feature columns named consistently. If a table contains a 'saved_path',
  it is ignored for modeling.

Quick debug run (first 20 IDs):
    python src/models/train_bernoulli_nb.py --hu_csv data/processed/2d_hu/hu_features_table.csv --pca_csv data/processed/descriptors/descriptors_table_pca.csv \
        --labels_csv data/splits/train.csv --features both  --binarize_method median  --val_size 0.2 \
        --seed 42 --fast

Full run with CV and alpha grid:
    python train_bernoulli_nb.py --hu_csv data/processed/2d_hu/hu_features_table.csv --pca_csv data/processed/descriptors/descriptors_table_pca.csv --labels_csv data/splits/train.csv  --features both --binarize_method quantile --quantile 0.5 --cv_folds 5 --alpha_grid 0.1,0.5,1.0 --seed 42

Final test evaluation (after training):
    python src/models/train_bernoulli_nb.py --... --test_csv data/splits/test.csv





----------
python -m train_bernoulli_nb `
  --raw_dir ../data/raw/dataverse_files `
  --test_dir ../data/raw/Test `
  --hu_csv ../data/processed/2d_hu/hu_features_table.csv `
  --pca_csv ../data/processed/descriptors/descriptors_table_pca.csv `
  --dry_run


"""
from __future__ import annotations
import re

import argparse
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    yaml = None  # type: ignore
    _HAS_YAML = False

try:
    from skimage.filters import threshold_otsu  # type: ignore
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_bernoulli_nb")

# -------------------------
# I/O Helpers
# -------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_table(path: Path, required_cols: Optional[List[str]] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    df = pd.read_csv(path)
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"File {path} missing required columns: {missing}")
    return df


def parse_alpha_grid(alpha_grid: Optional[str]) -> Optional[List[float]]:
    if not alpha_grid:
        return None
    try:
        values = [float(a.strip()) for a in alpha_grid.split(",") if a.strip()]
        return values if values else None
    except Exception:
        logger.warning("Failed to parse --alpha_grid; ignoring.")
        return None


# -------------------------
# NEW: Directory-based labels support
# -------------------------
def build_labels_from_dir(
    dir_path: Path,
    exts: Optional[Tuple[str, ...]] = None,
    id_format: str = "stem",
) -> pd.DataFrame:
    """
    Walk a directory with per-class subfolders and return a DataFrame with columns ['id','label'].
    - dir_path/class_name/**/*.ext
    - id_format:
        * 'stem' -> id = filename stem
        * 'class_stem' -> id = f"{class}_{stem}"
    - label is the immediate subfolder name under dir_path
    - Raises ValueError on duplicates of id across classes
    """
    if exts is None:
        exts = (".png", ".jpg", ".jpeg", ".npy", ".ply", ".pcd", ".txt", ".csv")
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    rows: List[Tuple[str, str]] = []
    classes = [d for d in sorted(dir_path.iterdir(), key=lambda p: p.name) if d.is_dir()]
    if not classes:
        raise ValueError(f"No class subfolders found under {dir_path}")
    for cls_dir in classes:
        label = cls_dir.name
        # recurse under class folder
        for p in sorted(cls_dir.rglob('*')):
            if p.is_file() and p.suffix.lower() in exts:
                stem = p.stem
                if id_format == "class_stem":
                    sid = f"{label}_{stem}"
                else:
                    sid = stem
                rows.append((sid, label))
    if not rows:
        raise ValueError(f"No files with supported extensions under {dir_path}")
    df = pd.DataFrame(rows, columns=["id", "label"]).drop_duplicates()
    # Detect duplicates of id mapping to multiple labels
    dup_counts = df["id"].value_counts()
    dups = dup_counts[dup_counts > 1].index.tolist()
    if dups:
        examples = dups[:5]
        raise ValueError(f"Duplicate ids across classes detected (e.g., {examples}). Ensure unique filename stems per sample.")
    # Deterministic ordering
    df = df.sort_values("id").reset_index(drop=True)
    return df


def coerce_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure an 'id' column exists (lowercase). Accepts 'Id' or 'ID'. Returns a new dataframe with 'id' column."""
    if "id" in df.columns:
        return df
    for cand in ["Id", "ID"]:
        if cand in df.columns:
            df = df.rename(columns={cand: "id"})
            return df
    return df


def coerce_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a 'label' column exists (lowercase). Accepts 'Label' or 'LABEL'."""
    if "label" in df.columns:
        return df
    for cand in ["Label", "LABEL"]:
        if cand in df.columns:
            df = df.rename(columns={cand: "label"})
            return df
    return df

# -------------------------
# Feature handling
# -------------------------

def select_feature_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c not in ("id", "label", "saved_path")]  # keep numeric features
    # further filter to numeric only
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return num_cols


def merge_features(
    hu_df: Optional[pd.DataFrame],
    pca_df: Optional[pd.DataFrame],
    features_choice: str,
) -> pd.DataFrame:
    """Merge selected features on 'id'. Returns DataFrame with 'id' and feature columns."""
    if features_choice not in {"hu", "pca", "both"}:
        raise ValueError("--features must be one of {'hu','pca','both'}")

    chosen: List[pd.DataFrame] = []
    if features_choice in ("hu", "both") and hu_df is not None:
        cols = ["id"] + select_feature_columns(hu_df)
        chosen.append(hu_df[cols].copy())
    if features_choice in ("pca", "both") and pca_df is not None:
        cols = ["id"] + select_feature_columns(pca_df)
        chosen.append(pca_df[cols].copy())

    if not chosen:
        raise ValueError("No feature tables available for the chosen --features option.")

    df = chosen[0]
    for nxt in chosen[1:]:
        df = pd.merge(df, nxt, on="id", how="outer", suffixes=("", "_pca"))
    return df


def align_with_labels(features_df: pd.DataFrame, labels_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Align features with labels by 'id'. Drop ids missing in features with warning."""
    if "id" not in features_df.columns:
        raise ValueError("Features DataFrame must contain 'id' column")
    if not {"id", "label"}.issubset(labels_df.columns):
        raise ValueError("Labels DataFrame must contain 'id' and 'label' columns")

    # report missing ids
    f_ids = set(features_df["id"].astype(str))
    l_ids = set(labels_df["id"].astype(str))
    missing_in_features = sorted(l_ids - f_ids)
    if missing_in_features:
        logger.warning(f"{len(missing_in_features)} ids from labels missing in features. They will be skipped. Example: {missing_in_features[:5]}")

    # inner-join to keep labeled samples that have features
    merged = pd.merge(labels_df[["id", "label"]], features_df, on="id", how="inner")

    # report feature rows without labels (not used)
    missing_in_labels = sorted(f_ids - l_ids)
    if missing_in_labels:
        logger.info(f"{len(missing_in_labels)} feature ids have no label and will be ignored for training.")

    y = merged["label"].astype(str)
    X = merged.drop(columns=["label"])  # still contains 'id'
    return X, y


def drop_and_impute(X_train: pd.DataFrame, X_other: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Drop all-NaN rows/cols in TRAIN, apply same column filter to OTHER. Impute NaNs with training column median."""
    # Identify feature columns (exclude id)
    feat_cols = [c for c in X_train.columns if c != "id"]
    sub_train = X_train[feat_cols]

    # Drop all-NaN columns in train
    keep_cols = [c for c in sub_train.columns if not sub_train[c].isna().all()]
    if len(keep_cols) < len(sub_train.columns):
        dropped = sorted(set(sub_train.columns) - set(keep_cols))
        logger.warning(f"Dropping {len(dropped)} all-NaN feature columns: {dropped[:10]}{'...' if len(dropped)>10 else ''}")
    X_train = pd.concat([X_train[["id"]], sub_train[keep_cols]], axis=1)

    # Drop all-NaN rows in train (rare)
    sub_train2 = X_train[keep_cols]
    mask_all_nan_rows = sub_train2.isna().all(axis=1)
    if mask_all_nan_rows.any():
        logger.warning(f"Dropping {int(mask_all_nan_rows.sum())} rows with all-NaN features from training set.")
        X_train = X_train.loc[~mask_all_nan_rows].reset_index(drop=True)

    # Apply same column selection to other
    if X_other is not None:
        X_other = pd.concat([X_other[["id"]], X_other[keep_cols]], axis=1)

    # Impute with training medians
    med = X_train[keep_cols].median(axis=0, skipna=True)
    X_train[keep_cols] = X_train[keep_cols].fillna(med)
    if X_other is not None:
        X_other[keep_cols] = X_other[keep_cols].fillna(med)

    return X_train, X_other


# -------------------------
# Thresholding / Binarization
# -------------------------

def _otsu_threshold_1d(values: np.ndarray) -> float:
    """Otsu threshold for 1D array. Uses skimage if available, else numpy fallback."""
    if values.size == 0:
        return 0.0
    vals = values.astype(float)
    if _HAS_SKIMAGE:
        try:
            return float(threshold_otsu(vals))
        except Exception:
            pass
    # fallback: histogram-based Otsu
    hist, bin_edges = np.histogram(vals, bins=256)
    hist = hist.astype(float)
    p = hist / hist.sum() if hist.sum() > 0 else hist
    omega = np.cumsum(p)
    mu = np.cumsum(p * ((bin_edges[:-1] + bin_edges[1:]) / 2.0))
    mu_t = mu[-1] if len(mu) else 0.0
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    idx = np.nanargmax(sigma_b2) if sigma_b2.size else 0
    thr = (bin_edges[idx] + bin_edges[idx + 1]) / 2.0 if idx + 1 < len(bin_edges) else bin_edges[idx]
    return float(thr)


def fit_thresholds(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "median",
    quantile: float = 0.5,
    strategy: str = "global",
) -> Dict[str, float]:
    """Compute per-feature thresholds based on training set only.
    - method: 'median'|'quantile'|'otsu'
    - strategy: 'global' (single threshold per feature) or 'per_class' (mean of class-wise thresholds)
    Returns {feature_name: threshold}.
    """
    assert strategy in {"global", "per_class"}
    feat_cols = [c for c in X_train.columns if c != "id"]

    thresholds: Dict[str, float] = {}

    def col_thresh(col_values: np.ndarray) -> float:
        if method == "median":
            return float(np.median(col_values))
        elif method == "quantile":
            return float(np.quantile(col_values, quantile))
        elif method == "otsu":
            return _otsu_threshold_1d(col_values)
        else:
            raise ValueError("Unknown binarize method")

    for c in feat_cols:
        values = X_train[c].astype(float).values
        if np.all(values == values[0]):
            # constant feature: set threshold to that constant; resulting column may be all 0 or all 1
            thresholds[c] = float(values[0])
            logger.warning(f"Feature '{c}' is constant in training. Threshold set to {thresholds[c]:.6g}")
            continue
        if strategy == "global":
            thresholds[c] = col_thresh(values)
        else:
            # per_class: compute threshold per class then take mean
            class_thrs: List[float] = []
            for cls in sorted(set(y_train)):
                cls_vals = X_train.loc[y_train == cls, c].astype(float).values
                if cls_vals.size == 0:
                    continue
                class_thrs.append(col_thresh(cls_vals))
            thresholds[c] = float(np.mean(class_thrs)) if class_thrs else col_thresh(values)
    return thresholds


def apply_thresholds(X: pd.DataFrame, thresholds: Dict[str, float]) -> np.ndarray:
    """Apply per-feature thresholds to DataFrame -> binary numpy array of shape (n_samples, n_features) with {0,1} uint8."""
    feat_cols = [c for c in X.columns if c != "id"]
    Xv = X[feat_cols].astype(float).values
    thr = np.array([thresholds[c] for c in feat_cols], dtype=float)
    bin_mat = (Xv >= thr).astype(np.uint8)
    return bin_mat


def save_thresholds_yaml(path: Path, thresholds: Dict[str, float], meta: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    payload = {
        "meta": meta,
        "thresholds": {k: float(v) for k, v in thresholds.items()},
    }
    if _HAS_YAML:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f)  # type: ignore
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, indent=2))


def load_thresholds_yaml(path: Path) -> Tuple[Dict[str, float], Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        if _HAS_YAML:
            data = yaml.safe_load(f)  # type: ignore
        else:
            data = json.loads(f.read())
    return {k: float(v) for k, v in data.get("thresholds", {}).items()}, data.get("meta", {})


# -------------------------
# Metrics & plotting
# -------------------------

def compute_metrics(y_true: Iterable[Any], y_pred: Iterable[Any], labels: Optional[List[Any]] = None) -> Dict[str, Any]:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    result: Dict[str, Any] = {}
    result["accuracy"] = float(accuracy_score(y_true, y_pred))
    result["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    result["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
    # per-class
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    if labels is None:
        labels = sorted(list(set(y_true)))
    for cls, p, r, f, s in zip(labels, prec, rec, f1, support):
        result[f"precision_{cls}"] = float(p)
        result[f"recall_{cls}"] = float(r)
        result[f"f1_{cls}"] = float(f)
        result[f"support_{cls}"] = int(s)
    # confusion
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    result["confusion_matrix"] = cm.tolist()
    result["labels"] = list(labels)
    return result


def plot_confusion_matrix(cm: np.ndarray, labels: List[Any], title: str, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=labels, yticklabels=labels,
           ylabel="True label", xlabel="Predicted label", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# -------------------------
# Training / CV
# -------------------------

def choose_cv(y: pd.Series, cv_folds: int, seed: int) -> Optional[StratifiedKFold]:
    if cv_folds is None or cv_folds <= 1:
        return None
    # verify min class count
    vc = y.value_counts()
    if (vc < cv_folds).any():
        logger.warning("Some classes have fewer samples than cv_folds. Falling back to StratifiedShuffleSplit for alpha selection.")
        return None
    return StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)


def alpha_selection(
    Xb: np.ndarray,
    y: pd.Series,
    alpha_default: float,
    alpha_grid: Optional[List[float]],
    cv: Optional[StratifiedKFold],
    seed: int,
) -> float:
    if not alpha_grid:
        return alpha_default
    best_alpha = alpha_default
    best_score = -np.inf
    # If cv is None due to small classes, use StratifiedShuffleSplit
    if cv is None:
        splitter = StratifiedShuffleSplit(n_splits=min(5, max(2, len(np.unique(y)))), test_size=0.2, random_state=seed)
        for a in alpha_grid:
            scores: List[float] = []
            for tr_idx, va_idx in splitter.split(Xb, y):
                clf = BernoulliNB(alpha=a)
                clf.fit(Xb[tr_idx], y.iloc[tr_idx])
                pred = clf.predict(Xb[va_idx])
                scores.append(f1_score(y.iloc[va_idx], pred, average="macro"))
            m = float(np.mean(scores))
            if m > best_score:
                best_score, best_alpha = m, a
        logger.info(f"Selected alpha={best_alpha} via StratifiedShuffleSplit (macro F1={best_score:.4f})")
        return best_alpha
    # Use KFold CV
    for a in alpha_grid:
        clf = BernoulliNB(alpha=a)
        scores = cross_val_score(clf, Xb, y, cv=cv, scoring="f1_macro")
        m = float(np.mean(scores))
        if m > best_score:
            best_score, best_alpha = m, a
    logger.info(f"Selected alpha={best_alpha} via {cv.get_n_splits()}-fold CV (macro F1={best_score:.4f})")
    return best_alpha


# -------------------------
# CLI
# -------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bernoulli Naive Bayes baseline with training-set binarization thresholds.")
    p.add_argument("--hu_csv", type=str, default="../data/processed/2d_hu/hu_features_table.csv")
    p.add_argument("--pca_csv", type=str, default="../data/processed/descriptors/descriptors_table_pca.csv")
    p.add_argument("--labels_csv", type=str, default="../data/splits/train.csv")
    p.add_argument("--test_csv", type=str, default=None)
    # New directory-based labels options (fallback if labels_csv missing)
    p.add_argument("--raw_dir", type=str, default="../data/raw/dataverse_files",
                   help="Directory with per-class subfolders for training samples (id=filename stem)")
    p.add_argument("--test_dir", type=str, default="data/raw/Test",
                   help="Optional directory with per-class subfolders for test samples; used if test_csv not provided")
    p.add_argument("--id_format", type=str, choices=["stem", "class_stem"], default="class_stem",
                   help="How to construct IDs when deriving labels from directories.")
    p.add_argument("--save_derived_splits", action="store_true",
                   help="If deriving labels from directories, save CSVs under derived_splits_outdir")
    p.add_argument("--derived_splits_outdir", type=str, default="../data/splits",
                   help="Output directory to save derived labels CSVs and unmatched ids list")
    p.add_argument("--dry_run", action="store_true", help="Print dataset summary (counts/intersections) and exit")

    p.add_argument("--features", type=str, choices=["hu", "pca", "both"], default=None,
                   help="Feature set to use. Default is 'both' if both files exist, else the one that exists.")

    p.add_argument("--binarize_method", type=str, choices=["median", "quantile", "otsu"], default="median")
    p.add_argument("--quantile", type=float, default=0.5)
    p.add_argument("--binarize_strategy", type=str, choices=["global", "per_class"], default="global")

    p.add_argument("--zscore", action="store_true", help="Optionally z-score features before binarization.")

    p.add_argument("--train_val_split", action="store_true", help="Use stratified holdout split for validation.")
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--cv_folds", type=int, default=0, help="If >1, run stratified K-fold CV for reporting and alpha selection.")

    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--alpha_grid", type=str, default=None, help="Comma-separated alpha values for grid search.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fast", action="store_true", help="Process only first 20 labeled samples and write outputs to debug folder.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Base directory to save outputs. If provided, experiments/results will be created under this path.")
    p.add_argument("--exp_dir", type=str, default=None,
    
    
                   help="Optional explicit experiments dir (overrides --out_dir).")
    p.add_argument("--res_dir", type=str, default=None,
                   help="Optional explicit results dir (overrides --out_dir).")

    return p


# -------------------------
# Main workflow
# -------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    rng = np.random.RandomState(args.seed)

    # Resolve defaults and existence
    hu_path = Path(args.hu_csv)
    pca_path = Path(args.pca_csv)
    labels_path = Path(args.labels_csv) if args.labels_csv else None
    test_path = Path(args.test_csv) if args.test_csv else None

    # Load labels (CSV preferred)
    labels_df: Optional[pd.DataFrame] = None
    test_labels_df: Optional[pd.DataFrame] = None
    if labels_path and labels_path.exists():
        labels_df = load_table(labels_path)
        labels_df = coerce_id_column(labels_df)
        labels_df = coerce_label_column(labels_df)
        if not {"id", "label"}.issubset(labels_df.columns):
            raise ValueError(f"Labels CSV {labels_path} must contain columns 'id' and 'label'")
        logger.info(f"Loaded labels from CSV: {labels_path} ({len(labels_df)} rows)")
    else:
        # Fallback to directory-based labels
        raw_dir = Path(args.raw_dir) if args.raw_dir else None
        if raw_dir is None or not raw_dir.exists():
            raise SystemExit("No labels source. Provide --labels_csv or a valid --raw_dir with class subfolders.")
        labels_df = build_labels_from_dir(raw_dir, id_format=str(args.id_format))
        logger.info(f"Derived {len(labels_df)} train labels from directory: {raw_dir}")
        if args.save_derived_splits:
            outdir = Path(args.derived_splits_outdir)
            ensure_dir(outdir)
            out_csv = outdir / "derived_train_labels.csv"
            labels_df.to_csv(out_csv, index=False)
            logger.info(f"Saved derived train labels to {out_csv}")
        # Optional test labels from directory if test_csv not provided
        test_dir = Path(args.test_dir) if args.test_dir else None
        if test_path is None and test_dir and test_dir.exists():
            try:
                test_labels_df = build_labels_from_dir(test_dir, id_format=str(args.id_format))
                if args.save_derived_splits:
                    outdir = Path(args.derived_splits_outdir)
                    ensure_dir(outdir)
                    out_csv_t = outdir / "derived_test_labels.csv"
                    test_labels_df.to_csv(out_csv_t, index=False)
                    logger.info(f"Saved derived test labels to {out_csv_t}")
            except Exception as e:
                logger.warning(f"Failed to derive test labels from {test_dir}: {e}")
    # If test_csv provided, attempt to load it; may or may not contain 'label'
    if test_path is not None and test_path.exists():
        test_labels_df = load_table(test_path)
        test_labels_df = coerce_id_column(test_labels_df)
        if "label" in test_labels_df.columns or any(c in test_labels_df.columns for c in ["Label", "LABEL"]):
            test_labels_df = coerce_label_column(test_labels_df)

    # Load features
    hu_df = None
    pca_df = None
    if hu_path.exists():
        hu_df = load_table(hu_path)
        hu_df = coerce_id_column(hu_df)
    else:
        logger.warning(f"HU features file not found at {hu_path}")
    if pca_path.exists():
        pca_df = load_table(pca_path)
        pca_df = coerce_id_column(pca_df)
    else:
        logger.warning(f"PCA features file not found at {pca_path}")

    features_choice = args.features
    if features_choice is None:
        if hu_df is not None and pca_df is not None:
            features_choice = "both"
        elif hu_df is not None:
            features_choice = "hu"
        elif pca_df is not None:
            features_choice = "pca"
        else:
            raise FileNotFoundError("Neither HU nor PCA features are available.")
    if features_choice == "both" and (hu_df is None or pca_df is None):
        logger.warning("--features=both requested but one feature file is missing. Falling back to available one.")
        features_choice = "hu" if hu_df is not None else "pca"

    features_df = merge_features(hu_df, pca_df, features_choice)

    # --- Normalize IDs to improve matching (strip extensions, replace separators/spaces) ---
    def _normalize_id(s: object) -> str:
        s = str(s)
        # remove common extensions
        s = re.sub(r'\.(pts|xyz|ply|pcd|npy|txt|csv)$', '', s, flags=re.IGNORECASE)
        s = s.replace('\\', '/')
        # convert directory separators to underscore (Class/123 -> Class_123)
        s = s.replace('/', '_')
        # replace spaces, colons, dashes with underscore
        s = re.sub(r'[\s\-\:]+', '_', s)
        # collapse multiple underscores
        s = re.sub(r'_+', '_', s)
        return s.strip('_')
    
    # Align with labels using intersection and log unmatched ids (with normalization)
    features_df = coerce_id_column(features_df)
    labels_df = coerce_id_column(labels_df)  # ensure labels_df has 'id' too
    
    # normalize both sides (safe even if already normalized)
    features_df['id'] = features_df['id'].astype(str).map(_normalize_id)
    labels_df['id'] = labels_df['id'].astype(str).map(_normalize_id)
    
    all_feat_ids = set(features_df["id"].astype(str))
    all_label_ids = set(labels_df["id"].astype(str))
    inter_ids = sorted(all_feat_ids & all_label_ids)
    dropped_from_labels = sorted(all_label_ids - all_feat_ids)
    dropped_from_features = sorted(all_feat_ids - all_label_ids)
    
    logger.info(f"IDs: features={len(all_feat_ids)}, labels={len(all_label_ids)}, intersection={len(inter_ids)}")
    if len(inter_ids) == 0:
        try:
            samp_feat = list(sorted(all_feat_ids))[:5]
            samp_label = list(sorted(all_label_ids))[:5]
            logger.warning("Intersection is zero. Sample feature IDs: %s", samp_feat)
            logger.warning("Sample label IDs: %s", samp_label)
            logger.warning("Tip: if your features use class-prefixed IDs like 'Buche_103', run with --id_format class_stem.")
        except Exception:
            pass
    if dropped_from_labels:
        logger.warning(f"Dropping {len(dropped_from_labels)} labeled ids missing features. Example: {dropped_from_labels[:5]}")
    if dropped_from_features:
        logger.info(f"{len(dropped_from_features)} feature ids have no label and will be ignored.")
    # Save unmatched ids list
        # make derived_splits_outdir follow out_dir if user didn't override it explicitly
    if args.out_dir and (args.derived_splits_outdir == "../data/splits" or args.derived_splits_outdir is None):
        unmatched_outdir = Path(args.out_dir) / "splits"
    else:
        unmatched_outdir = Path(args.derived_splits_outdir)
    ensure_dir(unmatched_outdir)

    with open(unmatched_outdir / "unmatched_ids.txt", "w", encoding="utf-8") as fh:
        fh.write("# In labels but not in features\n")
        for i in dropped_from_labels:
            fh.write(str(i) + "\n")
        fh.write("\n# In features but not in labels\n")
        for i in dropped_from_features:
            fh.write(str(i) + "\n")

    # Build aligned train tables
    labels_df = labels_df[labels_df["id"].astype(str).isin(inter_ids)].copy()
    features_df = features_df[features_df["id"].astype(str).isin(inter_ids)].copy()
    X, y = align_with_labels(features_df, labels_df)

    # Optional dry run
    if args.dry_run:
        logger.info("Dry run: summary only. Exiting.")
        return 0

        # Optional fast mode -> stratified subsample (safer than head(20))
    def stratified_subsample(X_df: pd.DataFrame, y_ser: pd.Series, n_samples: int, seed: int):
        """
        Return (X_sub, y_sub) as stratified subset of up to n_samples.
        Falls back to a small balanced/random selection if stratified split is impossible.
        """
        total = len(X_df)
        n_samples = min(int(n_samples), total)
        if n_samples <= 0:
            return X_df.copy(), y_ser.copy()

        tmp = X_df.reset_index(drop=True).copy()
        tmp["_label_tmp_"] = y_ser.reset_index(drop=True).astype(str).values

        try:
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, test_size=float(n_samples) / float(max(total, 1)), random_state=int(seed))
            for _, idx in sss.split(tmp, tmp["_label_tmp_"]):
                sel_idx = idx
            sel = tmp.iloc[sel_idx].reset_index(drop=True)
            Xs = sel.drop(columns=["_label_tmp_"])
            ys = sel["_label_tmp_"].astype(str).reset_index(drop=True)
            # If resulting sample has only one class, fall back
            if len(ys.unique()) <= 1:
                raise ValueError("Stratified split produced single-class subset; falling back.")
            return Xs, ys
        except Exception:
            # Fallback: ensure at least one sample per class if possible, then fill randomly
            rng = np.random.RandomState(seed)
            labels = list(tmp["_label_tmp_"].unique())
            chosen_idx = []
            # ensure at least one per class (up to budget)
            for lbl in labels:
                if len(chosen_idx) >= n_samples:
                    break
                cand = tmp[tmp["_label_tmp_"] == lbl].index.tolist()
                if cand:
                    chosen_idx.append(int(rng.choice(cand)))
            # fill remaining randomly
            remaining = [i for i in tmp.index.tolist() if i not in chosen_idx]
            rng.shuffle(remaining)
            while len(chosen_idx) < n_samples and remaining:
                chosen_idx.append(remaining.pop())
            # unique preserving order
            chosen_idx = list(dict.fromkeys(chosen_idx))
            sel = tmp.loc[chosen_idx].reset_index(drop=True)
            Xs = sel.drop(columns=["_label_tmp_"])
            ys = sel["_label_tmp_"].astype(str).reset_index(drop=True)
            if len(ys) == 0:
                return X_df.head(n_samples).copy(), y_ser.head(n_samples).copy()
            return Xs, ys

    if args.fast:
        logger.info("Fast mode: building a small stratified subset (up to 20 samples) for a quick debug run.")
        X_sub, y_sub = stratified_subsample(X.copy(), y.copy(), 20, int(args.seed))
        if len(set(y_sub.tolist())) <= 1:
            logger.warning("Fast subset contains only one label after stratified sampling; falling back to first 20 rows.")
            X = X.head(20).copy()
            y = y.head(20).copy()
        else:
            X = X_sub.copy()
            y = y_sub.copy()


    # Prepare output directories
        # -----------------------------
    # Output directories (configurable)
    # -----------------------------
    # priority: explicit exp_dir/res_dir > out_dir > defaults
    if args.exp_dir:
        base_exp_dir = Path(args.exp_dir)
    elif args.out_dir:
        base_exp_dir = Path(args.out_dir) / "experiments" / "bernoulli"
    else:
        base_exp_dir = Path("experiments") / "bernoulli"

    if args.res_dir:
        base_res_dir = Path(args.res_dir)
    elif args.out_dir:
        base_res_dir = Path(args.out_dir) / "results" / "bernoulli"
    else:
        base_res_dir = Path("results") / "bernoulli"

    # debug subfolder for fast mode (timestamped)
    if args.fast:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_exp_dir = base_exp_dir / f"debug_{stamp}"
        base_res_dir = base_res_dir / f"debug_{stamp}"

    ensure_dir(base_exp_dir)
    ensure_dir(base_res_dir)

    # Train/Val split or CV
    do_cv = args.cv_folds and args.cv_folds > 1
    do_split = args.train_val_split or not do_cv  # default to holdout if neither specified

    # Drop and impute NaNs using TRAIN information only. We'll first decide indices.
    indices_train = None
    indices_val = None

    if do_split:
        # If stratified split requested, check minimum class count first.
        # Stratify requires at least 2 samples per class (since test_size > 0).
        min_count = int(y.value_counts().min()) if len(y) > 0 else 0
        stratify_arg = y if min_count >= 2 else None
        if stratify_arg is None:
            logger.warning(
                "Cannot do stratified train/val split because some classes have <2 samples. "
                "Falling back to non-stratified random split."
            )
        try:
            X_train_df, X_val_df, y_train, y_val = train_test_split(
                X, y, test_size=args.val_size, random_state=args.seed, stratify=stratify_arg
            )
        except ValueError as e:
            # Defensive fallback: if sklearn still complains, perform non-stratified split
            logger.warning(f"Stratified split failed ({e}). Falling back to non-stratified split.")
            X_train_df, X_val_df, y_train, y_val = train_test_split(
                X, y, test_size=args.val_size, random_state=args.seed, stratify=None
            )

        indices_train = X_train_df["id"].tolist()
        indices_val = X_val_df["id"].tolist()
        # Drop/Impute based on training
        X_train_df, X_val_df = drop_and_impute(X_train_df, X_val_df)

    else:
        # CV will use all data; create single table with imputation based on full data median as proxy,
        # but ensure thresholds for binarization are computed within each CV split using training folds only.
        # Here we only need a cleaned table (no all-NaN cols/rows) using the entire data for feasibility.
        X_full_clean, _ = drop_and_impute(X.copy(), None)
        X_train_df, y_train = X_full_clean, y.copy()
        X_val_df, y_val = None, None

    # Optional z-score before binarization (computed on training set only)
    scaler: Optional[StandardScaler] = None
    if args.zscore:
        scaler = StandardScaler()
        feat_cols = [c for c in X_train_df.columns if c != "id"]
        scaler.fit(X_train_df[feat_cols].values)
        X_train_df[feat_cols] = scaler.transform(X_train_df[feat_cols].values)
        if X_val_df is not None:
            X_val_df[feat_cols] = scaler.transform(X_val_df[feat_cols].values)

    # --- Thresholds (fit on training set only) ---
    # Avant de calculer les seuils, on dÃƒÂ©tecte et enlÃƒÂ¨ve les colonnes constantes
    feat_cols = [c for c in X_train_df.columns if c != "id"]
    n_unique = X_train_df[feat_cols].nunique(dropna=False)
    
    constant_cols = [c for c in feat_cols if n_unique[c] <= 1]
    if constant_cols:
        logger.warning(
            f"Ignoring {len(constant_cols)} constant features with no variance: {constant_cols[:10]}{'...' if len(constant_cols) > 10 else ''}"
        )
        # on les enlÃƒÂ¨ve de train (et val si existe)
        X_train_df = X_train_df.drop(columns=constant_cols)
        if X_val_df is not None:
            X_val_df = X_val_df.drop(columns=[c for c in constant_cols if c in X_val_df.columns])
    
    # Maintenant calcul des seuils sur features restantes
    thresholds = fit_thresholds(
        X_train_df,
        y_train,
        method=args.binarize_method,
        quantile=args.quantile,
        strategy=args.binarize_strategy,
    )
    
    thr_meta = {
        "method": args.binarize_method,
        "strategy": args.binarize_strategy,
        "quantile": float(args.quantile),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "seed": int(args.seed),
        "features": features_choice,
        "zscore": bool(args.zscore),
    }
    save_thresholds_yaml(base_exp_dir / "bin_thresholds.yaml", thresholds, thr_meta)

    # Apply thresholds
    Xb_train = apply_thresholds(X_train_df, thresholds)
    # build canonical list of labels to use in confusion matrices
    if X_val_df is not None:
        # include labels present in train+val so CM squares include all labels seen
        labels_list = sorted(list(set(pd.concat([y_train, y_val]))))
    else:
        labels_list = sorted(list(set(y_train)))


    # Alpha selection (optional grid)
    cv_obj = choose_cv(y_train, int(args.cv_folds or 0), args.seed) if do_cv else None
    selected_alpha = alpha_selection(Xb_train, y_train, args.alpha, parse_alpha_grid(args.alpha_grid), cv_obj, args.seed)

    # Train model
    clf = BernoulliNB(alpha=selected_alpha)
    clf.fit(Xb_train, y_train)

    metrics_rows: List[Dict[str, Any]] = []

    # Validation evaluation if split
    if X_val_df is not None:
        Xb_val = apply_thresholds(X_val_df, thresholds)
        y_val_pred = clf.predict(Xb_val)
        m_val = compute_metrics(y_val, y_val_pred, labels=labels_list)
        m_val["split"] = "val"
        metrics_rows.append(m_val)
        # Save confusion matrix figure
        cm = np.array(m_val["confusion_matrix"], dtype=int)
        plot_confusion_matrix(cm, labels_list, title="Confusion Matrix - Val", out_path=base_res_dir / "confusion_matrix_val.png")
        # Save split indices
        pd.Series(indices_train, name="id").to_csv(base_exp_dir / "indices_train.csv", index=False)
        pd.Series(indices_val, name="id").to_csv(base_exp_dir / "indices_val.csv", index=False)

    # CV reporting (optional)
    if do_cv and cv_obj is not None:
        # Within each fold, refit thresholds on train fold only, then evaluate
        fold_scores: List[float] = []
        fold_acc: List[float] = []
        k = cv_obj.get_n_splits()
        for fold, (tr_idx, va_idx) in enumerate(cv_obj.split(X_train_df, y_train), start=1):
            Xtr = X_train_df.iloc[tr_idx].copy()
            Xva = X_train_df.iloc[va_idx].copy()
            ytr = y_train.iloc[tr_idx]
            yva = y_train.iloc[va_idx]
            # z-score using train fold only
            if args.zscore:
                sc = StandardScaler()
                fcols = [c for c in Xtr.columns if c != "id"]
                sc.fit(Xtr[fcols].values)
                Xtr[fcols] = sc.transform(Xtr[fcols].values)
                Xva[fcols] = sc.transform(Xva[fcols].values)
            thr_f = fit_thresholds(Xtr, ytr, method=args.binarize_method, quantile=args.quantile, strategy=args.binarize_strategy)
            Xb_tr = apply_thresholds(Xtr, thr_f)
            Xb_va = apply_thresholds(Xva, thr_f)
            clf_f = BernoulliNB(alpha=selected_alpha)
            clf_f.fit(Xb_tr, ytr)
            pred_va = clf_f.predict(Xb_va)
            f1m = f1_score(yva, pred_va, average="macro")
            acc = accuracy_score(yva, pred_va)
            fold_scores.append(f1m)
            fold_acc.append(acc)
        logger.info(f"CV ({k}-fold) macro F1: mean={np.mean(fold_scores):.4f} std={np.std(fold_scores):.4f}; accuracy mean={np.mean(fold_acc):.4f}")
        # --- AFTER CV loop: build OOF confusion matrix and save it to results dir ---
        try:
            # oof_preds array aligned with X_train_df rows (positional indices)
            oof_preds = np.empty(len(X_train_df), dtype=object)
            for fold, (tr_idx, va_idx) in enumerate(cv_obj.split(X_train_df, y_train), start=1):
                Xtr = X_train_df.iloc[tr_idx].copy()
                Xva = X_train_df.iloc[va_idx].copy()
                ytr = y_train.iloc[tr_idx]
                # fit thresholds on fold / binarize / train
                thr_f = fit_thresholds(Xtr, ytr, method=args.binarize_method, quantile=args.quantile, strategy=args.binarize_strategy)
                Xb_tr = apply_thresholds(Xtr, thr_f)
                Xb_va = apply_thresholds(Xva, thr_f)
                clf_f = BernoulliNB(alpha=selected_alpha)
                clf_f.fit(Xb_tr, ytr)
                pred_va = clf_f.predict(Xb_va)
                # store OOF preds into positional array
                oof_preds[va_idx] = pred_va
        
            # compute metrics and write confusion matrix for OOF preds
            if np.any([p is not None for p in oof_preds]):
                oof_preds_list = list(oof_preds)
                m_oof = compute_metrics(y_train, oof_preds_list, labels=labels_list)
                # append to metrics_rows
                m_oof["split"] = "oof_cv"
                metrics_rows.append(m_oof)
                # save confusion matrix figure
                cm_oof = np.array(m_oof["confusion_matrix"], dtype=int)
                plot_confusion_matrix(cm_oof, labels_list, title="Confusion Matrix - OOF CV", out_path=base_res_dir / "confusion_matrix_oof_cv.png")
                logger.info(f"Saved OOF CV confusion matrix to {base_res_dir / 'confusion_matrix_oof_cv.png'}")
        except Exception as e:
            logger.warning(f"Failed to build/save OOF confusion matrix: {e}")

        metrics_rows.append({
            "split": f"cv_{k}fold",
            "f1_macro_mean": float(np.mean(fold_scores)),
            "f1_macro_std": float(np.std(fold_scores)),
            "accuracy_mean": float(np.mean(fold_acc)),
            "accuracy_std": float(np.std(fold_acc)),
        })

    # Retrain final model on all available training data (train+val if split)
    if X_val_df is not None:
        X_all = pd.concat([X_train_df, X_val_df], axis=0, ignore_index=True)
        y_all = pd.concat([y_train, y_val], axis=0, ignore_index=True)
    else:
        X_all = X_train_df
        y_all = y_train

    # If zscore, refit scaler on all
    if args.zscore:
        scaler_all = StandardScaler()
        fcols_all = [c for c in X_all.columns if c != "id"]
        scaler_all.fit(X_all[fcols_all].values)
        X_all[fcols_all] = scaler_all.transform(X_all[fcols_all].values)
        # Refit thresholds on all training data
        thresholds_final = fit_thresholds(X_all, y_all, method=args.binarize_method, quantile=args.quantile, strategy=args.binarize_strategy)
    else:
        thresholds_final = fit_thresholds(X_all, y_all, method=args.binarize_method, quantile=args.quantile, strategy=args.binarize_strategy)

    save_thresholds_yaml(base_exp_dir / "bin_thresholds.yaml", thresholds_final, thr_meta)

    Xb_all = apply_thresholds(X_all, thresholds_final)
    final_clf = BernoulliNB(alpha=selected_alpha)
    final_clf.fit(Xb_all, y_all)

    # Save model
    ensure_dir(base_exp_dir)
    joblib.dump(final_clf, base_exp_dir / "bernoulli_model.joblib")

    # Test evaluation if provided via CSV or derived directory labels
    if test_path is not None and test_path.exists() or (test_labels_df is not None and len(test_labels_df) > 0):
        if test_path is not None and test_path.exists():
            test_ids_df = coerce_id_column(load_table(test_path))
            if "label" in test_ids_df.columns:
                test_ids_df = coerce_label_column(test_ids_df)
        else:
            test_ids_df = test_labels_df.copy()  # has id and label

        # Align with available features
        # Use the same feature columns as X_all
        keep_cols = ["id"] + [c for c in X_all.columns if c != "id"]
        test_feats = pd.merge(test_ids_df[["id"]], features_df, on="id", how="left")
        missing = test_feats["id"][test_feats.isna().all(axis=1)].tolist()
        if missing:
            logger.warning(f"{len(missing)} test ids missing features; they will be dropped from evaluation.")
        test_feats = test_feats.dropna(how="all")
        test_feats = test_feats[[c for c in keep_cols if c in test_feats.columns]].copy()
        # Impute using median of X_all
        med_all = X_all.drop(columns=["id"]).median(axis=0, skipna=True)
        test_feats = test_feats.fillna(med_all)
        # Z-score if used
        if args.zscore:
            scaler_all = StandardScaler()
            fcols_all = [c for c in X_all.columns if c != "id"]
            scaler_all.fit(X_all[fcols_all].values)
            test_feats[fcols_all] = scaler_all.transform(test_feats[fcols_all].values)
        # Apply thresholds
        Xb_test = apply_thresholds(test_feats, thresholds_final)
        # If labels available, compute metrics; else save predictions
        if "label" in test_ids_df.columns:
            # Align y_test order to test_feats ids
            y_test = test_ids_df.set_index("id").loc[test_feats["id"], "label"].astype(str)
            y_test_pred = final_clf.predict(Xb_test)
            m_test = compute_metrics(y_test, y_test_pred, labels=labels_list)
            m_test["split"] = "test"
            metrics_rows.append(m_test)
            cm = np.array(m_test["confusion_matrix"], dtype=int)
            plot_confusion_matrix(cm, labels_list, title="Confusion Matrix - Test", out_path=base_res_dir / "confusion_matrix_test.png")
        else:
            preds_out = pd.DataFrame({"id": test_feats["id"].tolist(), "pred": final_clf.predict(Xb_test)})
            preds_out.to_csv(base_exp_dir / "test_predictions.csv", index=False)
            logger.info("No test labels provided. Saved predictions only.")

    # Save metrics CSV
    if metrics_rows:
        # Flatten per-class metrics to columns; confusion matrix kept as JSON string
        flat_rows: List[Dict[str, Any]] = []
        for r in metrics_rows:
            rr = {k: v for k, v in r.items() if k not in ("confusion_matrix", "labels")}
            if "confusion_matrix" in r:
                rr["confusion_matrix_json"] = json.dumps({"labels": r.get("labels"), "matrix": r.get("confusion_matrix")})
            flat_rows.append(rr)
        pd.DataFrame(flat_rows).to_csv(base_exp_dir / "metrics.csv", index=False)

    # Save config YAML
    cfg = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "selected_features": features_choice,
        "alpha": float(selected_alpha),
    }
    ensure_dir(base_exp_dir)
    if _HAS_YAML:
        with open(base_exp_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)  # type: ignore
    else:
        with open(base_exp_dir / "config.yaml", "w", encoding="utf-8") as f:
            f.write(json.dumps(cfg, indent=2))

    # Save summary.txt
    summary_lines = []
    summary_lines.append(f"BernoulliNB training completed at {cfg['timestamp']}")
    summary_lines.append(f"Features used: {features_choice}")
    summary_lines.append(f"Alpha: {selected_alpha}")
    if metrics_rows:
        for r in metrics_rows:
            if r.get("split") in ("val", "test"):
                summary_lines.append(f"Split={r['split']}: acc={r.get('accuracy', 'n/a'):.4f} bal_acc={r.get('balanced_accuracy', 'n/a'):.4f} f1_macro={r.get('f1_macro', 'n/a'):.4f}")
            elif "cv_" in str(r.get("split", "")):
                summary_lines.append(f"{r['split']}: f1_macro_mean={r.get('f1_macro_mean', 'n/a'):.4f}Ã‚Â±{r.get('f1_macro_std', 'n/a'):.4f}")
    with open(base_exp_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    # Final stdout
    logger.info("Training finished. Outputs saved under:")
    logger.info(f"  - Model: {base_exp_dir / 'bernoulli_model.joblib'}")
    logger.info(f"  - Thresholds: {base_exp_dir / 'bin_thresholds.yaml'}")
    logger.info(f"  - Metrics: {base_exp_dir / 'metrics.csv'}")
    logger.info(f"  - Config: {base_exp_dir / 'config.yaml'}")
    logger.info(f"  - Confusion matrices: {base_res_dir}")
    if indices_train is not None:
        logger.info(f"  - Indices (train/val): {base_exp_dir}")
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        # Fast debug default
        demo_argv = [
            "--hu_csv", "../data/processed/2d_hu/hu_features_table.csv",
            "--pca_csv", "../data/processed/descriptors/descriptors_table_pca.csv",
            "--labels_csv", "../data/splits/train.csv",
            "--features", "both",
            "--binarize_method", "median",
            "--val_size", "0.2",
            "--seed", "42",
            "--fast",
            "--train_val_split",
        ]
        raise SystemExit(main(demo_argv))
    else:
        raise SystemExit(main())
