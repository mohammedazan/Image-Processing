#!/usr/bin/env python3
"""
eval.py — Unified evaluation script for classification models.

Usage (debug / FAST mode):
    python eval.py --model_dir ../experiments --test_csv ../data/raw/test.csv   --base_dir ../data/processed/sampled --output_dir ../reports --fast

Full run example:
    python eval.py --model_dir ../experiments --test_csv ../data/raw/test.csv  --base_dir ../data/processed/sampled --features_dir ../data/features  --output_dir ../reports

Notes:
- The script scans subdirectories under --model_dir; each subdir is expected to contain
  a model checkpoint (.pth or .joblib/.pkl). It will try to load an optional
  "label_encoder.joblib" in the same folder to map numeric predictions -> class names.
- For classical models (joblib/pkl), a features source is required via --features_dir
  (features.csv or features.npy). For pointnet/cnn, provide --base_dir so relative paths
  in test CSV resolve correctly.
- Test CSV formats supported:
  - path,class  (columns: "path","class")
  - id,label   (columns: "id","label" OR "id","class")
"""

from __future__ import annotations
import os
import sys
import argparse
import logging
import time
import json
from pathlib import Path
from typing import List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

# Try importing torch
try:
    import torch
    from torch import nn
except Exception:
    torch = None

# Optional dependency for otsu etc (not required here)
# from skimage.filters import threshold_otsu

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
logger = logging.getLogger("eval")

# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # deterministic behaviour (may slow down)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# -------------------------
# Test CSV loading
# -------------------------
def load_test_labels(test_csv: str) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    Return: (ids_list, labels_list, dataframe)
    Supports:
      - columns: path,class  (path -> id derived from basename, label in 'class')
      - columns: id,label   (or id,class or id,species)
    """
    df = pd.read_csv(test_csv)
    # Case A: path + class
    if "path" in df.columns and "class" in df.columns:
        df = df.copy()
        df["id"] = df["path"].apply(lambda x: os.path.splitext(os.path.basename(str(x)))[0])
        df = df.rename(columns={"class": "label"})
        ids = df["id"].astype(str).tolist()
        labels = df["label"].astype(str).tolist()
        return ids, labels, df
    # Case B: id + label-like
    id_cols = [c for c in df.columns if c.lower() == "id"]
    label_cols = [c for c in df.columns if c.lower() in ("label", "class", "species")]
    if id_cols and label_cols:
        id_col = id_cols[0]
        label_col = label_cols[0]
        ids = df[id_col].astype(str).tolist()
        labels = df[label_col].astype(str).tolist()
        return ids, labels, df
    raise ValueError(f"Unsupported test CSV format. Found columns: {list(df.columns)}")

# -------------------------
# Feature loading helper (for classical models)
# -------------------------
def load_features_for_ids(features_dir: str, ids: List[str]) -> np.ndarray:
    """
    Load features for ids. Supports:
      - features.csv (first column named 'ID' or 'id')
      - features.npy (aligned order)
    Returns: numpy array shape (len(ids), n_features)
    """
    fdir = Path(features_dir)
    if not fdir.exists():
        raise FileNotFoundError(f"features_dir {features_dir} not found")

    # Try CSV
    csv_paths = [p for p in fdir.glob("*.csv")]
    if (fdir / "features.csv").exists():
        csv_path = fdir / "features.csv"
    elif (fdir / "features_table.csv").exists():
        csv_path = fdir / "features_table.csv"
    elif csv_paths:
        csv_path = csv_paths[0]
    else:
        csv_path = None

    if csv_path is not None and csv_path.exists():
        df = pd.read_csv(csv_path, dtype=object)
        # find ID-like column
        id_col = None
        for cand in ("ID", "id", "Id", "sample_id"):
            if cand in df.columns:
                id_col = cand
                break
        if id_col is None:
            # assume first column is id
            id_col = df.columns[0]
        df_indexed = df.set_index(id_col)
        # convert remaining columns to numeric matrix
        feat_df = df_indexed.apply(pd.to_numeric, errors="coerce")
        feats = []
        for sid in ids:
            if sid in feat_df.index:
                feats.append(feat_df.loc[sid].values.astype(np.float32))
            else:
                logger.warning(f"ID {sid} not found in features CSV ({csv_path}). Using zeros.")
                feats.append(np.zeros(feat_df.shape[1], dtype=np.float32))
        return np.vstack(feats)

    # Try npy
    npy_path = fdir / "features.npy"
    if npy_path.exists():
        arr = np.load(str(npy_path))
        if arr.shape[0] != len(ids):
            logger.warning("features.npy rows != number of ids. Returning arr anyway if shape fits.")
        return arr

    # Fallback: per-id npy files
    perid = {}
    for p in sorted(fdir.glob("*.npy")):
        perid[p.stem] = np.load(str(p))
    if perid:
        feats = []
        # find feature length
        first = next(iter(perid.values()))
        nfeat = first.shape[0] if first.ndim == 1 else first.shape[1]
        for sid in ids:
            if sid in perid:
                v = perid[sid]
                if v.ndim > 1 and v.shape[0] == 1:
                    v = v.ravel()
                feats.append(np.asarray(v, dtype=np.float32).ravel())
            else:
                logger.warning(f"ID {sid} not found in per-id npy features. Using zeros.")
                feats.append(np.zeros(nfeat, dtype=np.float32))
        return np.vstack(feats)

    raise FileNotFoundError("No features file (CSV/NPY/per-id) found in features_dir")

# -------------------------
# Point cloud loader for PointNet
# -------------------------
def load_and_preprocess_pointcloud(file_path: str, num_points: int = 1024) -> torch.Tensor:
    """
    Load a point cloud from .npy/.txt/.xyz/.pts and return tensor (1, 3, N)
    """
    if torch is None:
        raise RuntimeError("torch is required to evaluate pointnet models")
    import numpy as _np

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    if file_path.endswith(".npy"):
        pts = _np.load(file_path).astype(_np.float32)
    else:
        # fallback to whitespace-delimited loader
        pts = _np.loadtxt(file_path, dtype=_np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"Unexpected pointcloud shape: {pts.shape}")
    pts = pts[:, :3]
    # sampling / pad
    N = pts.shape[0]
    if N >= num_points:
        idx = np.random.choice(N, num_points, replace=False)
    else:
        idx = np.random.choice(N, num_points, replace=True)
    sampled = pts[idx, :].astype(np.float32)
    # to tensor shape (1, 3, N)
    tensor = torch.tensor(sampled.T, dtype=torch.float32).unsqueeze(0)
    return tensor

# -------------------------
# Model evaluation (per-model)
# -------------------------
def evaluate_model(
    model: Any,
    model_type: str,
    test_ids: List[str],
    test_df: pd.DataFrame,
    features_dir: Optional[str],
    base_dir: Optional[str],
    device: Optional[torch.device],
) -> List[Any]:
    """
    Return list of predictions aligned with test_ids. Predictions can be:
      - class name (str)
      - class index (int)
      - -1 or None for failed/unavailable
    """
    preds = []
    if model_type in ("bernoulli", "stacking", "sklearn"):
        if not features_dir:
            raise ValueError("features_dir is required for classical models")
        X_test = load_features_for_ids(features_dir, test_ids)
        preds = model.predict(X_test).tolist()
        return preds

    if model_type == "pointnet":
        if torch is None:
            raise RuntimeError("PyTorch not available to run PointNet evaluation")
        model.to(device)
        model.eval()
        preds = []
        with torch.no_grad():
            for idx, row in test_df.iterrows():
                rel = row.get("path") or row.get("id")
                if not rel or (isinstance(rel, float) and np.isnan(rel)):
                    preds.append(-1)
                    continue
                file_path = os.path.join(base_dir, rel) if base_dir else rel
                try:
                    data = load_and_preprocess_pointcloud(file_path)
                    data = data.to(device)
                    out = model(data)  # expecting (1, n_classes)
                    if hasattr(out, "detach"):
                        out = out.detach()
                    # logits/probs -> argmax
                    if isinstance(out, torch.Tensor):
                        pred_idx = int(torch.argmax(out, dim=1).cpu().item())
                    else:
                        # numpy array
                        pred_idx = int(np.argmax(out, axis=1).item())
                    preds.append(pred_idx)
                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {e}")
                    preds.append(-1)
        return preds

    # default: try sklearn predict
    try:
        if features_dir:
            X_test = load_features_for_ids(features_dir, test_ids)
            return model.predict(X_test).tolist()
    except Exception:
        pass

    # fallback: cannot evaluate
    raise NotImplementedError(f"Evaluation not implemented for model_type={model_type}")

# -------------------------
# Helpers: mapping predictions -> class names (strings)
# -------------------------
def load_label_encoder_if_any(model_dir: str):
    le_path = Path(model_dir) / "label_encoder.joblib"
    if le_path.exists():
        try:
            le = joblib.load(str(le_path))
            logger.info(f"Loaded LabelEncoder from {le_path}")
            return le
        except Exception as e:
            logger.warning(f"Failed to load label encoder {le_path}: {e}")
            return None
    return None

def preds_to_classnames(preds_raw: List[Any], le: Optional[Any], global_classes: List[str]) -> List[Optional[str]]:
    """
    Convert predictions (ints or strings) to class name strings when possible.
    Returns list of strings or None for missing/unmapped.
    Priority:
      - If pred is None or -1 -> None
      - If pred is string -> return as-is
      - If pred is int and le provided -> le.inverse_transform
      - If pred is int and le missing -> try map to global_classes by index
    """
    out = []
    for p in preds_raw:
        if p is None or p == -1:
            out.append(None)
            continue
        if isinstance(p, str):
            out.append(p)
            continue
        # numeric -> try to map
        try:
            pi = int(p)
        except Exception:
            out.append(str(p))
            continue
        if le is not None:
            try:
                cname = str(le.inverse_transform([pi])[0])
                out.append(cname)
            except Exception:
                logger.debug(f"Cannot inverse_transform {pi} with provided encoder")
                out.append(None)
        else:
            # fall back to mapping using global_classes ordering (best-effort)
            if 0 <= pi < len(global_classes):
                out.append(global_classes[pi])
            else:
                logger.warning(f"Predicted index {pi} out of range for global_classes (n={len(global_classes)})")
                out.append(None)
    return out

# -------------------------
# Metrics / plotting
# -------------------------
def compute_metrics(y_true: List[str], y_pred: List[Optional[str]]):
    """
    y_true: list of strings (labels)
    y_pred: list of strings or None
    Returns dict with metrics and dictionaries for precision/recall per class
    """
    # Filter invalid
    mask = [p is not None for p in y_pred]
    if not any(mask):
        logger.warning("No valid predictions available.")
        return None

    y_true_valid = [yt for yt, m in zip(y_true, mask) if m]
    y_pred_valid = [yp for yp, m in zip(y_pred, mask) if m]

    # Convert to arrays
    y_true_arr = np.array(y_true_valid)
    y_pred_arr = np.array(y_pred_valid)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    bal = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
    f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    report = classification_report(y_true_arr, y_pred_arr, output_dict=True, zero_division=0)
    precision = {}
    recall = {}
    for cls, vals in report.items():
        if cls in ("accuracy", "macro avg", "weighted avg"):
            continue
        precision[cls] = float(vals.get("precision", 0.0))
        recall[cls] = float(vals.get("recall", 0.0))
    return {"accuracy": acc, "balanced_accuracy": bal, "macro_f1": f1, "precision": precision, "recall": recall, "n_valid": int(sum(mask)), "n_total": len(y_pred)}

def save_metrics_and_confusion(y_true: List[str], y_pred: List[Optional[str]], metrics: dict, out_dir: str, model_name: str):
    p = Path(out_dir)
    ensure_dir(p)
    # save metrics.json
    metrics_path = p / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=4, ensure_ascii=False)
    logger.info(f"Saved metrics.json -> {metrics_path}")

    # Confusion matrix (only valid preds)
    mask = [p is not None for p in y_pred]
    y_true_valid = [yt for yt, m in zip(y_true, mask) if m]
    y_pred_valid = [yp for yp, m in zip(y_pred, mask) if m]
    if len(y_true_valid) == 0 or len(set(y_pred_valid)) == 0:
        logger.warning("No valid predictions for confusion matrix; skipping.")
        return

    classes = sorted(list(set(y_true_valid)))
    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=classes)
    # plot
    plt.figure(figsize=(max(6, len(classes) * 0.6), max(4, len(classes) * 0.6)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    cm_path = p / "confusion_matrix.png"
    plt.savefig(str(cm_path))
    plt.close()
    logger.info(f"Saved confusion matrix -> {cm_path}")

# -------------------------
# identify model type
# -------------------------
def identify_model_type(checkpoint_path: str, dir_name: str) -> str:
    ext = Path(checkpoint_path).suffix.lower()
    if ext in (".joblib", ".pkl"):
        # heuristic: folder name might contain 'bernoulli' or 'mv_stack'
        return "sklearn"
    if ext == ".pth":
        # guess by dirname
        if "pointnet" in dir_name.lower():
            return "pointnet"
        # else generic cnn (not implemented)
        return "pointnet"
    return "unknown"

# -------------------------
# Main
# -------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate trained models on test set (unified).")
    p.add_argument("--model_dir", type=str, default="experiments", help="Folder containing model subfolders")
    p.add_argument("--test_csv", type=str, required=True, help="Test CSV file (path,class OR id,label)")
    p.add_argument("--features_dir", type=str, default="data/features", help="Features dir for classical models (CSV/NPY/per-id)")
    p.add_argument("--base_dir", type=str, default=".", help="Base dir to resolve relative paths from test CSV (point clouds/images)")
    p.add_argument("--output_dir", type=str, default="results", help="Where to save metrics/confusion")
    p.add_argument("--fast", action="store_true", help="Fast mode: use only first 20 test samples (debug)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--verbose", action="store_true", help="Debug logging")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    set_seed(args.seed)

    logger.info("Starting evaluation")
    test_ids, y_true, test_df = load_test_labels(args.test_csv)
    logger.info(f"Loaded {len(test_ids)} test samples from {args.test_csv}")

    if args.fast:
        # Keep first 20 rows for quick debugging
        n = min(20, len(test_ids))
        logger.info(f"FAST eval mode: using first {n} samples of test set")
        test_ids = test_ids[:n]
        y_true = y_true[:n]
        test_df = test_df.iloc[:n].reset_index(drop=True)

    # discover model subfolders
    model_dirs = []
    md = Path(args.model_dir)
    if md.is_file():
        model_dirs.append(md.parent)
    elif md.is_dir():
        # each subdir corresponds to a model run
        for p in sorted(md.iterdir()):
            if p.is_dir():
                # check for any checkpoint inside
                have_ckpt = any([str(x).lower().endswith((".pth", ".joblib", ".pkl")) for x in p.iterdir()])
                if have_ckpt:
                    model_dirs.append(p)
            else:
                # if file directly in experiments folder
                if p.suffix.lower() in (".pth", ".joblib", ".pkl"):
                    model_dirs.append(md)
                    break
    else:
        logger.error(f"model_dir {args.model_dir} not found")
        sys.exit(1)

    if not model_dirs:
        logger.error("No model directories found to evaluate.")
        sys.exit(1)

    # Prepare aggregated CSV of results
    agg_rows = []
    ensure_dir(Path(args.output_dir))

    for model_path in model_dirs:
        model_name = model_path.name
        logger.info(f"Processing model directory: {model_path} (name={model_name})")

        # locate checkpoint file
        ckpt = None
        for ext in (".pth", ".joblib", ".pkl"):
            candidates = list(model_path.glob(f"*{ext}"))
            if candidates:
                ckpt = str(candidates[0])
                break
        if ckpt is None:
            logger.warning(f"No checkpoint file found in {model_path}. Skipping.")
            continue

        model_type = identify_model_type(ckpt, model_name)
        logger.info(f"Detected model type: {model_type}, checkpoint={ckpt}")

        # load label encoder if present
        le = load_label_encoder_if_any(str(model_path))

        # load model
        model = None
        device = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu") if torch is not None else None
        try:
            if ckpt.endswith((".joblib", ".pkl")):
                model = joblib.load(ckpt)
                logger.info(f"Loaded sklearn model from {ckpt}")
            elif ckpt.endswith(".pth") and torch is not None:
                # Expect a PointNet implementation in models.pointnet.PointNetCls or similar
                try:
                    from models.pointnet import PointNetCls  # user project module
                    # determine n_classes from test labels
                    classes_global = sorted(list(set(y_true)))
                    n_classes = max(2, len(classes_global))
                    logger.info(f"Rebuilding PointNet with {n_classes} classes")
                    model = PointNetCls(k=n_classes, feature_transform=False)
                except Exception as e:
                    logger.warning(f"Could not import PointNetCls from models.pointnet: {e}. Trying generic state_dict load onto model-less container.")
                    # If project doesn't contain class, try loading state_dict into a dummy nn.Module
                    class DummyNet(nn.Module):
                        def __init__(self):
                            super().__init__()
                        def forward(self, x):
                            raise RuntimeError("DummyNet cannot forward - missing model class")
                    model = DummyNet()
                # load checkpoint into model (non-strict)
                try:
                    ck = torch.load(ckpt, map_location=device)
                    if isinstance(ck, dict):
                        # find candidate keys
                        if "state_dict" in ck:
                            sd = ck["state_dict"]
                        elif "model_state_dict" in ck:
                            sd = ck["model_state_dict"]
                        else:
                            sd = ck
                    else:
                        sd = ck
                    # try to load (may warn)
                    if hasattr(model, "load_state_dict"):
                        missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
                        logger.debug(f"Loaded state_dict with missing={missing_keys}, unexpected={unexpected_keys}")
                    else:
                        logger.debug("Model instance has no load_state_dict; skipping state load.")
                except Exception as e:
                    logger.warning(f"Failed loading .pth checkpoint into model: {e}")
                model.to(device)
                model.eval()
            else:
                logger.error(f"Unsupported checkpoint format: {ckpt}. Skipping.")
                continue
        except Exception as e:
            logger.error(f"Failed to load model from {ckpt}: {e}")
            continue

        # Evaluate
        start = time.time()
        try:
            preds_raw = evaluate_model(model, model_type, test_ids, test_df, args.features_dir, args.base_dir, device)
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            continue
        elapsed = time.time() - start
        logger.info(f"Prediction time for {model_name}: {elapsed:.2f}s")

        # Map preds -> classnames (strings)
        global_classes = sorted(list(set(y_true)))
        preds_names = preds_to_classnames(preds_raw, le, global_classes)

        # Build metrics
        metrics = compute_metrics(y_true, preds_names)
        if metrics is None:
            logger.warning(f"No valid predictions for model {model_name}; saving partial info and continuing.")
            metrics = {"accuracy": None, "balanced_accuracy": None, "macro_f1": None, "precision": {}, "recall": {}, "n_valid": 0, "n_total": len(preds_names)}
        metrics["prediction_time"] = float(elapsed)
        metrics["model_name"] = model_name

        # Save per-model results
        out_dir_model = Path(args.output_dir) / model_name
        save_metrics_and_confusion(y_true, preds_names, metrics, str(out_dir_model), model_name)

        # add to aggregated CSV
        agg_row = {
            "model": model_name,
            "accuracy": metrics.get("accuracy"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "n_valid": metrics.get("n_valid"),
            "n_total": metrics.get("n_total"),
            "prediction_time": metrics.get("prediction_time"),
        }
        agg_rows.append(agg_row)

    # write aggregated metrics CSV
    if agg_rows:
        agg_df = pd.DataFrame(agg_rows)
        agg_csv = Path(args.output_dir) / "metrics.csv"
        agg_df.to_csv(str(agg_csv), index=False)
        logger.info(f"Wrote aggregated metrics to {agg_csv}")
    else:
        logger.info("No models evaluated; nothing to aggregate.")

    logger.info("Evaluation finished.")

if __name__ == "__main__":
    main()
