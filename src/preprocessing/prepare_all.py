"""

prepare_all.py

Preprocessing script for the "Tree Species Classification" project.

Usage examples:
  Debug / fast run:
    python prepare_all.py --raw_dir "Dataset/dataverse_files" --out_dir .../../../../data/processed --npoints 1024 --fast --workers 2
  Full run:
    python prepare_all.py --raw_dir "Dataset/dataverse_files" --out_dir .../../../../data/processed --npoints 1024 --workers 8

This script:
 - Reads point clouds from supported formats (.ply, .pcd, .xyz, .txt, .npy, .pts)
 - Performs statistical outlier removal (default nb_neighbors=20, std_ratio=2.0)
 - Centers points to centroid = (0,0,0)
 - Scales to unit sphere (max distance -> 1)
 - Estimates normals (k=16)
 - Applies Farthest Point Sampling (FPS) to produce exactly N points (upsamples deterministically if needed)
 - Saves outputs:
     data/processed/sampled/<id>.npy  (N x 3, float32)
     data/processed/normals/<id>.npy  (N x 3, float32)
   and appends a row to:
     data/processed/manifest.csv (id,status,sampled_path,normals_path,raw_path,notes)

Design notes / justifications:
 - Statistical outlier defaults (nb_neighbors=20, std_ratio=2.0) are conservative and commonly used for moderate-density scans.
 - FPS implemented in numpy to ensure deterministic behavior and avoid dependency on environment-specific Open3D functions.
 - Per-sample processing is done in worker processes; manifest updates happen in the main process to avoid concurrent writes.
 - Determinism: seeds are set for Python random and numpy; worker seeds are derived deterministically from the provided seed.
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import math
import zlib

import random
import numpy as np
import pandas as pd
from tqdm import tqdm

# Multiprocessing
from multiprocessing import Pool

# Try to import Open3D; if not available some features will fallback to numpy/text loaders.
try:
    import open3d as o3d
    _HAS_OPEN3D = True
except Exception:
    o3d = None
    _HAS_OPEN3D = False

# Constants / sensible defaults
DEFAULT_RAW_DIR = Path("../../data/raw/dataverse_files")
DEFAULT_OUT_DIR = Path(".../../../../data/processed")
DEFAULT_NPOINTS = 1024
DEFAULT_WORKERS = 4
DEFAULT_SEED = 42

# Statistical outlier removal defaults
SOR_NB_NEIGHBORS = 20
SOR_STD_RATIO = 2.0

# Normal estimation parameter
NORMAL_K = 16

# Supported extensions
SUPPORTED_EXTS = {".ply", ".pcd", ".xyz", ".txt", ".npy", ".pts"}

# Manifest columns
MANIFEST_COLUMNS = ["id", "status", "sampled_path", "normals_path", "raw_path", "notes"]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("prepare_all")


def read_pointcloud(path: Path) -> np.ndarray:
    """
    Read a pointcloud from path. Supports .ply, .pcd via Open3D when available,
    and text-based .xyz/.txt/.pts or .npy (numpy array) as fallback.

    Returns:
        points: (M,3) numpy float64 array.

    Raises:
        FileNotFoundError, ValueError
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(path, allow_pickle=False)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(f"Invalid .npy shape for point cloud: {arr.shape}")
        return arr[:, :3].astype(np.float64)

    # Try Open3D for common formats
    if _HAS_OPEN3D and ext in {".ply", ".pcd", ".xyz", ".pts"}:
        try:
            pcd = o3d.io.read_point_cloud(str(path))
            pts = np.asarray(pcd.points)
            if pts.size > 0:
                return pts.astype(np.float64)
            # otherwise fallback to text loader
        except Exception as e:
            logger.debug(f"Open3D failed to read {path}: {e}")

    # Fallback text loader: accept space or comma separated columns, skip non-numeric lines
    coords = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # support comma or whitespace separators
            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                continue
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                coords.append((x, y, z))
            except ValueError:
                continue
    if len(coords) == 0:
        raise ValueError(f"No valid XYZ rows found in {path}")
    return np.asarray(coords, dtype=np.float64)


def clean_pointcloud(points: np.ndarray,
                     nb_neighbors: int = SOR_NB_NEIGHBORS,
                     std_ratio: float = SOR_STD_RATIO) -> np.ndarray:
    """
    Apply statistical outlier removal (SOR). If Open3D is available, use its implementation
    which is efficient and standard. If not, perform a simple fallback that returns input.

    Parameters:
        points: (M,3) numpy array
        nb_neighbors: neighborhood size
        std_ratio: standard deviation multiplier

    Returns:
        filtered_points: numpy array (K,3)
    """
    if points.shape[0] == 0:
        return points
    if _HAS_OPEN3D:
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
            filtered = np.asarray(cl.points)
            if filtered.size == 0:
                # if filter removes everything, fallback to original
                logger.warning("SOR removed all points; returning original cloud")
                return points
            return filtered
        except Exception as e:
            logger.warning(f"Open3D SOR failed: {e}; returning original points")
            return points
    else:
        # Fallback: no SOR available; return original while logging.
        logger.debug("Open3D not available: skipping SOR; returning original points")
        return points


def center_and_scale(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Center points to centroid = (0,0,0) and scale to unit sphere (max distance -> 1).

    Returns:
        normalized_points: (M,3) float32
        centroid: (3,) float32
    """
    if points.size == 0:
        return points.astype(np.float32), np.zeros(3, dtype=np.float32)
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    dists = np.linalg.norm(centered, axis=1)
    max_dist = float(np.max(dists)) if dists.size > 0 else 0.0
    if max_dist <= 0:
        scaled = centered
    else:
        scaled = centered / max_dist
    return scaled.astype(np.float32), centroid.astype(np.float32)


def estimate_normals(points: np.ndarray, k: int = NORMAL_K) -> np.ndarray:
    """
    Estimate normals for given points. Prefer Open3D estimation (k-NN). If not available,
    return zero normals and log a warning.

    Returns:
        normals: (M,3) float32 array
    """
    M = points.shape[0]
    if M == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if _HAS_OPEN3D:
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
            normals = np.asarray(pcd.normals, dtype=np.float32)
            if normals.shape[0] != M:
                logger.warning("Estimated normals count differs from points; filling zeros")
                return np.zeros((M, 3), dtype=np.float32)
            return normals
        except Exception as e:
            logger.warning(f"Open3D normal estimation failed: {e}; returning zeros")
            return np.zeros((M, 3), dtype=np.float32)
    else:
        logger.warning("Open3D not available: cannot estimate normals; returning zeros")
        return np.zeros((M, 3), dtype=np.float32)


def farthest_point_sampling(points: np.ndarray, n_samples: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    Deterministic numpy implementation of FPS (Farthest Point Sampling).

    Parameters:
        points: (M,3) float32 or float64
        n_samples: desired number of samples
        rng: numpy RandomState for deterministic upsampling when M < n_samples

    Returns:
        indices: (n_samples,) np.int64 indices into points
    """
    M = points.shape[0]
    if M == 0:
        return np.array([], dtype=np.int64)
    if n_samples <= 0:
        return np.array([], dtype=np.int64)

    if n_samples >= M:
        # Upsample deterministically by repeating indices and shuffling with rng if provided
        reps = int(math.ceil(n_samples / M))
        base = np.tile(np.arange(M, dtype=np.int64), reps)[:n_samples]
        if rng is not None:
            rng.shuffle(base)
        return base.astype(np.int64)

    # Classic O(M * n_samples) FPS
    pts = points.astype(np.float64)
    indices = np.zeros(n_samples, dtype=np.int64)
    # start with a deterministic point: choose index 0 to be deterministic seedable alternative to random
    # but to add randomness we can use rng.choice if provided, else deterministic 0
    if rng is not None:
        start = int(rng.randint(0, M))
    else:
        start = 0
    indices[0] = start
    # distances to selected set
    dists = np.linalg.norm(pts - pts[start], axis=1)
    for i in range(1, n_samples):
        far_idx = int(np.argmax(dists))
        indices[i] = far_idx
        new_dists = np.linalg.norm(pts - pts[far_idx], axis=1)
        dists = np.minimum(dists, new_dists)
    return indices


def safe_save_npy(path: Path, arr: np.ndarray) -> None:
    """
    Save numpy array with float32 dtype and allow_pickle=False. Create parent dir if needed.
    """
    ensure_dir(path.parent)
    arr_to_save = arr.astype(np.float32, copy=False)
    np.save(str(path), arr_to_save, allow_pickle=False)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def gather_input_files(raw_dir: Path, exts: Optional[set] = None) -> List[Path]:
    """
    Recursively gather files under raw_dir matching supported extensions.
    """
    if exts is None:
        exts = SUPPORTED_EXTS
    files: List[Path] = []
    for p in raw_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    files_sorted = sorted(files)
    return files_sorted


def make_id_from_path(path: Path, raw_root: Path) -> str:
    """
    Create an id string from path relative to raw_root.
    Replace path separators with underscores to produce a filesystem-friendly id.
    """
    try:
        rel = path.relative_to(raw_root)
    except Exception:
        rel = path.name
    # remove suffix
    stem = str(rel.with_suffix(""))
    id_str = stem.replace(os.sep, "_").replace("/", "_")
    # sanitize spaces
    id_str = id_str.replace(" ", "_")
    return id_str


def process_single(path_str: str,
                   raw_root_str: str,
                   out_dir_str: str,
                   npoints: int,
                   overwrite: bool,
                   seed: int,
                   verbose: bool = False) -> Dict:
    """
    Process a single file path (string-friendly for multiprocessing).

    Returns a record dict for manifest:
      {'id': id, 'status': 'processed'|'skipped'|'error', 'sampled_path':..., 'normals_path':..., 'raw_path':..., 'notes':...}
    """
    path = Path(path_str)
    raw_root = Path(raw_root_str)
    out_dir = Path(out_dir_str)
    record = {
        "id": make_id_from_path(path, raw_root),
        "status": "error",
        "sampled_path": "",
        "normals_path": "",
        "raw_path": str(path.resolve()),
        "notes": ""
    }
    try:
        if not path.exists():
            record["status"] = "error"
            record["notes"] = f"File not found: {path}"
            return record

        id_name = record["id"]
        sampled_path = out_dir / "sampled" / f"{id_name}.npy"
        normals_path = out_dir / "normals" / f"{id_name}.npy"

        # Skip existing unless overwrite
        if sampled_path.exists() and normals_path.exists() and not overwrite:
            record.update({
                "status": "skipped",
                "sampled_path": str(sampled_path.resolve()),
                "normals_path": str(normals_path.resolve()),
                "notes": "already_exists"
            })
            return record

        # Read points
        points = read_pointcloud(path)

        if points.shape[0] < 3:
            record["status"] = "error"
            record["notes"] = f"Too few points ({points.shape[0]})"
            return record

        # Clean
        cleaned = clean_pointcloud(points, nb_neighbors=SOR_NB_NEIGHBORS, std_ratio=SOR_STD_RATIO)

        # Center & scale
        normalized, centroid = center_and_scale(cleaned)

        # Estimate normals (on normalized)
        normals_full = estimate_normals(normalized, k=NORMAL_K)

        # Prepare RNG for deterministic FPS and potential upsampling
        # Use numpy RandomState for deterministic behavior
        # Derive worker-unique seed using provided seed and stable path hash to avoid collisions
        path_hash = zlib.adler32(str(path).encode("utf-8"))  # stable across runs
        rng_seed = int((seed + int(path_hash)) % (2 ** 31 - 1))
        rng = np.random.RandomState(rng_seed)

        # FPS indices
        idxs = farthest_point_sampling(normalized.astype(np.float32), npoints, rng=rng)
        # If returned fewer than npoints (should not happen), pad by repeating last idx
        if idxs.size < npoints:
            pad = np.full((npoints - idxs.size,), idxs[-1] if idxs.size > 0 else 0, dtype=np.int64)
            idxs = np.concatenate([idxs, pad])

        sampled = normalized[idxs].astype(np.float32)
        sampled_normals = normals_full[idxs].astype(np.float32)

        # Save outputs
        safe_save_npy(sampled_path, sampled)
        safe_save_npy(normals_path, sampled_normals)

        record.update({
            "status": "processed",
            "sampled_path": str(sampled_path.resolve()),
            "normals_path": str(normals_path.resolve()),
            "notes": f"centroid:{centroid.tolist()},orig_points:{points.shape[0]},cleaned:{cleaned.shape[0]}"
        })
        if verbose:
            logger.info(f"Processed {path} -> {sampled_path} ({sampled.shape})")

        return record
    except Exception as e:
        tb = traceback.format_exc()
        record["status"] = "error"
        record["notes"] = f"{type(e).__name__}:{str(e)}"
        # include truncated traceback for diagnostics
        record["notes"] += f" | tb:{tb.splitlines()[-1] if tb else 'no_tb'}"
        logger.error(f"Error processing {path}: {e}")
        logger.debug(tb)
        return record


def write_manifest_row(manifest_path: Path, record: Dict) -> None:
    """
    Append a single record row to the manifest CSV. Creates file with header if missing.
    This function is intended to be called from the main process only (avoids concurrency issues).
    """
    ensure_dir(manifest_path.parent)
    file_exists = manifest_path.exists()
    # Use pandas to ensure safe csv writing semantics
    df = pd.DataFrame([record])
    if not file_exists:
        df.to_csv(manifest_path, index=False, columns=MANIFEST_COLUMNS)
    else:
        # Append without header
        df.to_csv(manifest_path, mode="a", index=False, header=False, columns=MANIFEST_COLUMNS)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare point clouds: denoise, center, scale, normals, FPS sampling.")
    parser.add_argument("--raw_dir", type=str, default=str(DEFAULT_RAW_DIR),
                        help=f"Directory with raw point cloud files (default: {DEFAULT_RAW_DIR})")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR),
                        help=f"Output base directory where sampled/ and normals/ will be written (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--npoints", type=int, default=DEFAULT_NPOINTS, help="Number of FPS points to sample (default 1024)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of parallel workers for processing (default 4)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--fast", action="store_true", help="Fast mode: process only first 10 samples for debugging")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility (default 42)")
    parser.add_argument("--verbose", action="store_true", help="Show extra logs")
    return parser.parse_args(argv)


def init_worker(seed: int) -> None:
    """
    Initializer for worker processes to set deterministic seeds.
    Each worker will use a seed derived from the base seed and its worker id.
    """
    try:
        from multiprocessing import current_process
        wid = 0
        ident = getattr(current_process(), "_identity", None)
        if ident:
            wid = ident[0]
        # Derive worker seed deterministically
        wseed = int((seed + wid) % (2 ** 31 - 1))
        random.seed(wseed)
        np.random.seed(wseed)
    except Exception:
        random.seed(seed)
        np.random.seed(seed)


def worker_wrapper(args_tuple):
    """
    Top-level worker wrapper so multiprocessing can pickle this function on Windows.
    Unpacks args and calls process_single.
    """
    return process_single(*args_tuple)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    npoints = int(args.npoints)
    workers = max(1, int(args.workers))
    overwrite = bool(args.overwrite)
    fast = bool(args.fast)
    seed = int(args.seed)
    verbose = bool(args.verbose)

    if verbose:
        logger.setLevel(logging.DEBUG)

    # Set base seeds
    random.seed(seed)
    np.random.seed(seed)

    if not raw_dir.exists():
        logger.error(f"raw_dir does not exist: {raw_dir}")
        return 2

    files = gather_input_files(raw_dir)
    if len(files) == 0:
        logger.error(f"No supported point cloud files found under {raw_dir}")
        return 3

    if fast:
        files = files[:10]

    total = len(files)
    logger.info(f"Found {total} files under {raw_dir}. Workers={workers}, npoints={npoints}, overwrite={overwrite}, fast={fast}")

    manifest_path = out_dir / "manifest.csv"
    ensure_dir(out_dir / "sampled")
    ensure_dir(out_dir / "normals")

    # Prepare worker pool
    records_processed = 0
    records_skipped = 0
    records_error = 0

    # We will call process_single in workers but manifest updates happen in main process.
    # Map arguments per file as tuple of strings so process_single signature is simple.
    tasks = [(str(p), str(raw_dir), str(out_dir), npoints, overwrite, seed, verbose) for p in files]

    # Use multiprocessing pool
    if workers > 1:
        with Pool(processes=workers, initializer=init_worker, initargs=(seed,)) as pool:
            # Use imap_unordered for responsive progress updates
            it = pool.imap_unordered(worker_wrapper, tasks)
            with tqdm(total=total, desc="Processing", unit="file") as pbar:
                for rec in it:
                    # Write manifest row atomically (main process)
                    write_manifest_row(manifest_path, rec)
                    status = rec.get("status", "").lower()
                    if status == "processed":
                        records_processed += 1
                    elif status == "skipped":
                        records_skipped += 1
                    else:
                        records_error += 1
                    pbar.update(1)
    else:
        # Single-process run (easier debugging)
        with tqdm(total=total, desc="Processing", unit="file") as pbar:
            for t in tasks:
                rec = worker_wrapper(t)
                write_manifest_row(manifest_path, rec)
                status = rec.get("status", "").lower()
                if status == "processed":
                    records_processed += 1
                elif status == "skipped":
                    records_skipped += 1
                else:
                    records_error += 1
                pbar.update(1)

    logger.info(f"Processing done: {records_processed} processed, {records_skipped} skipped, {records_error} errors. Manifest at {manifest_path}")
    return 0


if __name__ == "__main__":
    # If no CLI args provided, default to a safe unit test: process only first sample.
    if len(sys.argv) == 1:
        logger.info("No CLI args provided — running unit test on a single sample (safe mode).")
        # Build args for a fast single-sample run
        argv = [
            "--raw_dir", str(DEFAULT_RAW_DIR),
            "--out_dir", str(DEFAULT_OUT_DIR),
            "--npoints", str(DEFAULT_NPOINTS),
            "--workers", "1",
            "--fast",
            "--seed", str(DEFAULT_SEED)
        ]
        ret = main(argv)
        # Print verification summary for first processed sample (if any)
        manifest_p = DEFAULT_OUT_DIR / "manifest.csv"
        if manifest_p.exists():
            try:
                df = pd.read_csv(manifest_p)
                if not df.empty:
                    # pick last appended row
                    row = df.iloc[-1]
                    if row["status"] == "processed":
                        sampled_path = Path(row["sampled_path"])
                        normals_path = Path(row["normals_path"])
                        sampled = np.load(sampled_path, allow_pickle=False)
                        normals = np.load(normals_path, allow_pickle=False)
                        centroid = np.mean(sampled, axis=0)
                        print(f"Sample processed: id={row['id']}, sampled_shape={sampled.shape}, normals_shape={normals.shape}, centroid=({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})")
                    else:
                        print(f"Unit test finished: last manifest entry status={row['status']}, notes={row.get('notes','')}")
                else:
                    print("Unit test: manifest is empty.")
            except Exception as e:
                print(f"Unit test: failed to read manifest or sample files: {e}")
        else:
            print("Unit test: manifest not found.")
        sys.exit(ret)
    else:
        parsed_args = parse_args()
        sys.exit(main(sys.argv[1:]))