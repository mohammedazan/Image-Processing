#!/usr/bin/env python3
"""
extract_3d_pca_features.py

Task A3 - PCA-based 3D descriptor extraction for Tree Species Classification.

Usage examples:

Debug / fast mode:
    python extract_3d_pca_features.py --sampled_dir data/processed/sampled --out_dir data/processed/descriptors --fast

Full run:
    python extract_3d_pca_features.py --sampled_dir data/processed/sampled --out_dir data/processed/descriptors --n_bins 8 --workers 4

This script reads sampled point clouds (<id>.npy), computes PCA-based descriptors
(eigenvalues, ratios, projection histogram on PC1), saves per-sample descriptor
as data/processed/descriptors/pca_<id>.npy and a global CSV
data/processed/descriptors/descriptors_table_pca.csv. It also writes
reports/pca_feature_stats.csv summarizing mean/std per descriptor column.

Notes:
 - No binarization is performed here.
 - Deterministic given the same seed.
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path
import math
import datetime
import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pca_features")


# -----------------------
# Helper functions
# -----------------------
def load_sample(path: Path) -> np.ndarray:
    """
    Load a sampled point cloud from .npy file and return Nx3 float64 array.
    Raises exception if invalid.
    """
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"invalid shape {arr.shape} (expected Nx3) for {path}")
    return arr[:, :3].astype(np.float64)


def compute_pca_features(points: np.ndarray, n_bins: int = 8, normalize: bool = False) -> np.ndarray:
    """
    Compute PCA-based descriptor vector for points (Nx3).

    Returns a 1D float32 vector in order:
      [lambda1, lambda2, lambda3, r1, r2, r3, ratio12, ratio23, trace,
       proj_hist_bin_0, ..., proj_hist_bin_{n_bins-1}]
    - lambda1 >= lambda2 >= lambda3 (non-negative, clipped tiny negatives to 0)
    - r_i = lambda_i / trace (trace = sum lambdas)
    - ratio12 = lambda1 / (lambda2 + eps), ratio23 = lambda2 / (lambda3 + eps)
    - projection histogram: use absolute projection on v1, normalized counts (sum->1)
    If normalize=True, P_c is divided by its max norm (documented use).
    """
    if points.shape[0] < 3:
        raise ValueError("too few points (<3)")

    # Center
    centroid = points.mean(axis=0)
    P_c = points - centroid

    if normalize:
        norms = np.linalg.norm(P_c, axis=1)
        maxn = float(np.max(norms)) if norms.size > 0 else 0.0
        if maxn > 0:
            P_c = P_c / maxn

    # Covariance (3x3)
    # rowvar=False: variables are columns
    C = np.cov(P_c, rowvar=False, bias=False)
    # Ensure symmetric
    C = (C + C.T) / 2.0

    # Eigen-decomposition (use eigh for symmetric)
    evals, evecs = np.linalg.eigh(C)  # ascending order
    # Convert to descending
    evals = evals[::-1]
    evecs = evecs[:, ::-1]

    # Clip small negatives to zero
    eps_clip = 1e-12
    evals = np.where(evals < 0.0, np.clip(evals, a_min=0.0, a_max=None), evals)
    evals = evals.astype(float)

    lambda1, lambda2, lambda3 = float(evals[0]), float(evals[1]), float(evals[2])
    trace = lambda1 + lambda2 + lambda3
    trace_safe = trace if trace > eps_clip else eps_clip

    # ratios r1,r2,r3
    r1 = lambda1 / trace_safe
    r2 = lambda2 / trace_safe
    r3 = lambda3 / trace_safe

    # pairwise ratios with small eps to avoid div0
    ratio12 = lambda1 / (lambda2 + 1e-12)
    ratio23 = lambda2 / (lambda3 + 1e-12)

    # projection histogram on PC1: absolute projection to avoid sign ambiguity
    v1 = evecs[:, 0]
    proj = np.abs(P_c.dot(v1.astype(np.float64)))
    # define bins from 0..max_proj inclusive; handle zero variance
    max_proj = float(np.max(proj)) if proj.size > 0 else 0.0
    if max_proj <= 0:
        # all zeros -> histogram has all zeros except maybe first bin
        counts = np.zeros(n_bins, dtype=float)
        # put 1 in first bin to represent degenerate
        counts[0] = 1.0
    else:
        bins = np.linspace(0.0, max_proj, n_bins + 1, endpoint=True)
        counts, _ = np.histogram(proj, bins=bins)
        counts = counts.astype(float)
        total = counts.sum()
        if total > 0:
            counts = counts / total  # normalized histogram
        else:
            counts = np.zeros_like(counts, dtype=float)

    # Build feature vector in exact ordering
    feat_list = [
        lambda1, lambda2, lambda3,
        r1, r2, r3,
        ratio12, ratio23,
        trace
    ]
    feat_list.extend([float(x) for x in counts.tolist()])

    vec = np.asarray(feat_list, dtype=np.float32)
    return vec


def save_descriptor(sample_id: str, vec: np.ndarray, out_dir: Path) -> Path:
    """
    Save per-sample descriptor vector as pca_<id>.npy (dtype float32). Returns path.
    """
    ensure_dir(out_dir)
    fname = f"pca_{sample_id}.npy"
    out_path = out_dir / fname
    # np.save overwrites existing file by default
    np.save(str(out_path), vec.astype(np.float32), allow_pickle=False)
    return out_path


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def build_manifest_map(manifest_path: Path) -> dict:
    mapping = {}
    if manifest_path.exists():
        try:
            df = pd.read_csv(manifest_path)
            if "id" in df.columns and "raw_path" in df.columns:
                for _, r in df.iterrows():
                    mapping[str(r["id"])] = str(r["raw_path"])
        except Exception as e:
            logger.warning(f"Could not read manifest {manifest_path}: {e}")
    return mapping

# Add module-level worker initializer so it is picklable on Windows
def _init_worker(seed: int) -> None:
    """
    Initialize worker RNG deterministically. Must be at module scope so it is pickleable
    by multiprocessing on Windows (spawn).
    """
    try:
        from multiprocessing import current_process
        ident = getattr(current_process(), "_identity", None)
        wid = ident[0] if ident else 0
    except Exception:
        wid = 0
    np.random.seed((int(seed) + int(wid)) % (2 ** 31 - 1))


# -----------------------
# Worker/process function
# -----------------------
def process_file(args_tuple):
    """
    Worker-level processing function. Accepts a tuple of arguments to be pickle-friendly.
    Returns a dict record for CSV or an error dict.
    """
    (file_path_str, sampled_dir_str, out_dir_str, n_bins, normalize, overwrite, manifest_map_str, reports_dir_str) = args_tuple
    file_path = Path(file_path_str)
    sampled_dir = Path(sampled_dir_str)
    out_dir = Path(out_dir_str)
    reports_dir = Path(reports_dir_str)
    manifest_map = manifest_map_str if isinstance(manifest_map_str, dict) else {}

    record = {
        "id": file_path.stem,
        "raw_path": manifest_map.get(file_path.stem, ""),
        "n_points": None,
        "lambda1": None,
        "lambda2": None,
        "lambda3": None,
        "r1": None,   
        "r2": None,
        "r3": None,
        "ratio12": None,
        "ratio23": None,
        "trace": None,
        # proj hist bins will be filled later into flat list
        "proj_hist": None,
        "descriptor_path": ""
    }

    try:
        pts = load_sample(file_path)
        n_pts = int(pts.shape[0])
        if n_pts < 3:
            raise ValueError("too few points (<3) after loading")

        vec = compute_pca_features(pts, n_bins=n_bins, normalize=normalize)
        # save per-sample descriptor
        desc_path = save_descriptor(file_path.stem, vec, out_dir)
        record["descriptor_path"] = str(desc_path.resolve())
        # fill scalar fields and histogram
        # vec order: lambda1,lambda2,lambda3,r1,r2,r3,ratio12,ratio23,trace, bins...
        base_vals = vec[:9].astype(float).tolist()
        bins_vals = vec[9:].astype(float).tolist()
        (lambda1, lambda2, lambda3, r1, r2, r3, ratio12, ratio23, trace) = base_vals
        record.update({
            "n_points": n_pts,
            "lambda1": lambda1,
            "lambda2": lambda2,
            "lambda3": lambda3,
            "r1": r1,
            "r2": r2,
            "r3": r3,
            "ratio12": ratio12,
            "ratio23": ratio23,
            "trace": trace,
            "proj_hist": bins_vals
        })
        return {"success": True, "record": record, "error": ""}
    except Exception as e:
        # log to reports/read_errors.log
        try:
            ensure_dir(Path(reports_dir))
            with open(Path(reports_dir) / "read_errors.log", "a", encoding="utf-8") as fh:
                fh.write(f"{file_path}\t{type(e).__name__}\t{e}\n")
        except Exception:
            pass
        logger.debug(f"Error processing {file_path}: {e}")
        return {"success": False, "record": record, "error": str(e)}


# -----------------------
# Main orchestration
# -----------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Extract PCA-based 3D descriptors from sampled point clouds.")
    p.add_argument("--sampled_dir", type=str, default="../preprocessing/data/processed/sampled",
                   help="Directory with sampled .npy point clouds (default: ../preprocessing/data/processed/sampled)")
    p.add_argument("--out_dir", type=str, default="../preprocessing/data/processed/descriptors/",
                   help="Output base directory for descriptors (default: ../preprocessing/data/processed/descriptors/)")
    p.add_argument("--n_bins", type=int, default=8, help="Number of histogram bins for PC1 projections (default: 8)")
    p.add_argument("--max_samples", type=int, default=None, help="Limit number of samples processed (default: None)")
    p.add_argument("--fast", action="store_true", help="Fast debug shorthand (sets max_samples=20)")
    p.add_argument("--normalize", action="store_true",
                   help="Optional: normalize centered points by max norm before PCA (documented behaviour).")
    p.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default 1)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing per-sample descriptors and CSV")
    p.add_argument("--processed_manifest", type=str, default="",
                   help="Optional path to processed manifest CSV mapping id -> raw_path (default: ../manifest.csv relative to sampled_dir)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for deterministic behavior")
    p.add_argument("--verbose", action="store_true", help="Verbose logging (debug)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    np.random.seed(int(args.seed))

    sampled_dir = Path(args.sampled_dir)
    out_dir = Path(args.out_dir)
    reports_dir = Path("../preprocessing/reports/")
    ensure_dir(out_dir)
    ensure_dir(reports_dir)

    # manifest path resolution
    manifest_map = {}
    if args.processed_manifest:
        manifest_path = Path(args.processed_manifest)
        if manifest_path.exists():
            manifest_map = build_manifest_map(manifest_path)
    else:
        # default to sampled_dir/../manifest.csv
        candidate = sampled_dir.parent / "manifest.csv"
        if candidate.exists():
            manifest_map = build_manifest_map(candidate)

    if not sampled_dir.exists():
        logger.error(f"Sampled directory not found: {sampled_dir}")
        return 2

    # gather .npy files recursively
    files = sorted([p for p in sampled_dir.rglob("*.npy") if p.is_file()])
    # determinism: sort by path string
    files = sorted(files, key=lambda p: str(p))

    if args.fast:
        max_samples = 20
    else:
        max_samples = args.max_samples
    if max_samples is not None:
        files = files[:max_samples]

    total = len(files)
    logger.info(f"Found {total} sampled .npy files under {sampled_dir}. Workers={args.workers}, n_bins={args.n_bins}, normalize={args.normalize}, overwrite={args.overwrite}")

    if total == 0:
        logger.warning("No .npy files found; exiting.")
        return 0

    # Prepare worker args list
    tasks = []
    for p in files:
        tasks.append((str(p), str(sampled_dir), str(out_dir), int(args.n_bins), bool(args.normalize),
                      bool(args.overwrite), manifest_map, str(reports_dir)))

    results = []
    if args.workers > 1:
        # Use multiprocessing Pool
        worker_init_seed = int(args.seed)
        try:
            # NOTE: use module-level _init_worker (pickleable). Previously a nested function caused
            # "Can't get local object 'main.<locals>._init_worker'" on Windows.
            with mp.Pool(processes=args.workers, initializer=_init_worker, initargs=(worker_init_seed,)) as pool:
                for res in tqdm(pool.imap_unordered(process_file, tasks), total=len(tasks), desc="Processing"):
                    results.append(res)
        except Exception as e:
            logger.error(f"Multiprocessing failed: {e}. Falling back to single-process.")
            # fallback to single process
            for t in tqdm(tasks, desc="Processing (fallback)"):
                results.append(process_file(t))
    else:
        # single-process
        for t in tqdm(tasks, desc="Processing"):
            results.append(process_file(t))

    # Collect successful records and build CSV
    records = []
    n_processed = 0
    for r in results:
        if r.get("success"):
            rec = r["record"]
            # flatten proj_hist into individual columns later
            records.append(rec)
            n_processed += 1
        else:
            # log already written in worker; continue
            pass

    if len(records) == 0:
        logger.error("No descriptors produced (all files failed). Check reports/read_errors.log for details.")
        return 3

    # Build DataFrame with exact column order
    # Columns: id, raw_path, n_points, lambda1, lambda2, lambda3, r1, r2, r3, ratio12, ratio23, trace,
    # proj_hist_bin_0 ... proj_hist_bin_{n_bins-1}
    n_bins = int(args.n_bins)
    col_names = ["id", "raw_path", "n_points", "lambda1", "lambda2", "lambda3",
                 "r1", "r2", "r3", "ratio12", "ratio23", "trace"]
    bin_cols = [f"proj_hist_bin_{i}" for i in range(n_bins)]
    all_cols = col_names + bin_cols

    rows = []
    for rec in records:
        row = {
            "id": rec.get("id", ""),
            "raw_path": rec.get("raw_path", ""),
            "n_points": rec.get("n_points", np.nan),
            "lambda1": rec.get("lambda1", np.nan),
            "lambda2": rec.get("lambda2", np.nan),
            "lambda3": rec.get("lambda3", np.nan),
            "r1": rec.get("r1", np.nan),
            "r2": rec.get("r2", np.nan),
            "r3": rec.get("r3", np.nan),
            "ratio12": rec.get("ratio12", np.nan),
            "ratio23": rec.get("ratio23", np.nan),
            "trace": rec.get("trace", np.nan)
        }
        hist = rec.get("proj_hist", [0.0] * n_bins)
        # ensure length n_bins
        if hist is None:
            hist = [0.0] * n_bins
        if len(hist) < n_bins:
            hist = list(hist) + [0.0] * (n_bins - len(hist))
        for i in range(n_bins):
            row[f"proj_hist_bin_{i}"] = float(hist[i])
        rows.append(row)

    df_out = pd.DataFrame(rows, columns=all_cols)

    # Write CSV: descriptors_table_pca.csv under out_dir
    ensure_dir(out_dir)
    csv_path = out_dir / "descriptors_table_pca.csv"
    if csv_path.exists() and not args.overwrite:
        logger.info(f"{csv_path} exists and --overwrite not set. Overwriting is not requested; will overwrite anyway to ensure consistent table.")
    # Save CSV (overwrite if exists)
    df_out.to_csv(csv_path, index=False)
    logger.info(f"Wrote descriptors table to {csv_path}")

    # Save small summary stats to reports/pca_feature_stats.csv
    stats = df_out.select_dtypes(include=[np.number]).agg(["mean", "std"]).transpose().reset_index()
    stats.rename(columns={"index": "feature"}, inplace=True)
    stats_path = reports_dir / "pca_feature_stats.csv"
    stats.to_csv(stats_path, index=False)
    logger.info(f"Wrote feature stats to {stats_path}")

    # Print quick sanity check: first row
    try:
        first_row = df_out.iloc[0].to_dict()
        logger.info(f"First descriptor row: id={first_row.get('id')}, n_points={first_row.get('n_points')}")
    except Exception:
        pass

    mode = "fast" if args.fast else "full"
    print(f"PCA descriptor extraction finished. Mode: {mode}. Samples processed: {n_processed}. Descriptors saved to {out_dir}.")
    if csv_path.exists():
        try:
            df_preview = pd.read_csv(csv_path, nrows=1)
            print("First CSV row preview:")
            print(df_preview.to_dict(orient="records")[0])
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    # If invoked without args, run safe debug
    if len(sys.argv) == 1:
        argv = ["--sampled_dir", "data/processed/sampled", "--out_dir", "data/processed/descriptors", "--fast", "--workers", "1"]
        sys.exit(main(argv))
    else:
        sys.exit(main(sys.argv[1:]))