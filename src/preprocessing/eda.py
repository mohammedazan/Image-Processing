"""
eda.py

Exploratory Data Analysis for Tree Species Classification.

Usage examples:
  Debug (fast) using processed samples:
    python eda.py --processed_dir .../../../../data/processed/sampled --out_dir ../../reports --fast
  Full run on raw dataset:
    python eda.py --raw_dir ../../data/raw/dataverse_files --out_dir ../../reports  

This script automatically chooses processed mode if --processed_dir exists and contains .npy files;
otherwise it falls back to raw mode scanning --raw_dir recursively.

Outputs (exact paths):
 - reports/eda_summary.md
 - reports/class_counts.csv
 - reports/point_counts_per_file.csv
 - reports/sample_images/<class>_<id>_preview.png
 - reports/plots/point_count_histogram.png
 - reports/plots/class_balance_bar.png
 - reports/plots/points_boxplot_per_class.png
 - reports/read_errors.log

Notes:
 - Deterministic sampling: random.seed(seed) and numpy.random.seed(seed)
 - Reads .npy preferred; supports .ply/.pcd via Open3D if available; else text fallback parser.
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import datetime
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Optional Open3D
try:
    import open3d as o3d  # type: ignore
    _HAS_OPEN3D = True
except Exception:
    o3d = None
    _HAS_OPEN3D = False

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eda")

SUPPORTED_EXTS = {".npy", ".ply", ".pcd", ".xyz", ".txt", ".pts"}
DEFAULT_RAW_DIR = Path("../../data/raw/dataverse_files")
DEFAULT_PROCESSED_DIR = Path(".../../../../data/processed/sampled")
DEFAULT_OUT_DIR = Path("../../reports")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EDA for Tree Species Classification")
    p.add_argument("--raw_dir", type=str, default=str(DEFAULT_RAW_DIR),
                   help="Raw dataset root with class subfolders (default: ../../data/raw/dataverse_files)")
    p.add_argument("--processed_dir", type=str, default=str(DEFAULT_PROCESSED_DIR),
                   help="Directory with processed sampled .npy point clouds (preferred, default: .../../../../data/processed/sampled/)")
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory for reports (default: ../../reports/)")
    p.add_argument("--n_preview", type=int, default=3, help="Number of preview samples per class (default 3)")
    p.add_argument("--max_samples", type=int, default=None, help="Limit total number of samples to inspect")
    p.add_argument("--fast", action="store_true", help="Fast debug mode: restrict to at most 20 samples and smaller plots")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args(argv)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_npy(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if arr.ndim < 2 or arr.shape[1] < 3:
        raise ValueError(f".npy file does not contain Nx3 array: {path}")
    return arr[:, :3].astype(np.float32)


def read_with_open3d(path: Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        raise ValueError(f"Open3D returned empty point cloud for {path}")
    return pts.astype(np.float32)


def read_text_xyz(path: Path) -> np.ndarray:
    coords = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                continue
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                coords.append((x, y, z))
            except ValueError:
                continue
    if not coords:
        raise ValueError(f"No valid coordinates parsed from {path}")
    return np.asarray(coords, dtype=np.float32)


def load_pointcloud(path: Path) -> np.ndarray:
    """
    Load point cloud robustly:
      - .npy: load directly
      - if Open3D available and format supported: use it
      - fallback to text parser for whitespace/comma XYZ
    """
    ext = path.suffix.lower()
    if ext == ".npy":
        return read_npy(path)
    if _HAS_OPEN3D and ext in {".ply", ".pcd", ".xyz", ".pts"}:
        try:
            return read_with_open3d(path)
        except Exception as e:
            logger.debug(f"Open3D read failed for {path}: {e}; falling back to text parser")
    # fallback
    return read_text_xyz(path)


def sample_preview_views_matplotlib(points: np.ndarray, out_path: Path, size: Tuple[int, int] = (1024, 340)):
    """
    Create a 3-view collage (side-by-side) using simple 2D projections (XY, XZ, YZ) via matplotlib.
    """
    ensure_dir(out_path.parent)
    fig, axes = plt.subplots(1, 3, figsize=(size[0] / 100, size[1] / 100), dpi=100)
    axes_list = axes.ravel()
    projections = [(0, 1), (0, 2), (1, 2)]
    for ax, (i, j) in zip(axes_list, projections):
        ax.scatter(points[:, i], points[:, j], s=0.5, c="gray", alpha=0.6)
        ax.set_xlabel(["X", "Y", "Z"][i])
        ax.set_ylabel(["X", "Y", "Z"][j])
        ax.set_title(f"{['XY','XZ','YZ'][projections.index((i,j))]}")
        ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)


def render_preview_open3d(points: np.ndarray, out_path: Path, size: Tuple[int, int] = (1024, 340)):
    """
    Try to render 3 rotated views with Open3D's visualizer/OffscreenRenderer if available.
    If OffscreenRenderer is not available, fallback to matplotlib method.
    """
    # Offscreen rendering API availability varies; use simple fallback if not present
    try:
        ensure_dir(out_path.parent)
        vis_imgs = []
        # create temp geometry
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        # Use simple visualizer capture with different rotations
        # Attempt minimal approach: use geometry's rotate and capture via draw
        # Safer: for reliability, use matplotlib projections instead
        raise RuntimeError("Open3D offscreen not used: fallback to matplotlib")
    except Exception:
        sample_preview_views_matplotlib(points, out_path, size=size)


def compute_stats(points: np.ndarray) -> Dict:
    """
    Compute basic stats: n_points, bbox (min,max per axis), centroid, bounding volume, density.
    """
    n = int(points.shape[0])
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centroid = points.mean(axis=0)
    extents = maxs - mins
    # avoid zero volume
    volume = float(np.prod(extents.clip(min=1e-9)))
    density = n / volume if volume > 0 else float("inf")
    return {
        "n_points": n,
        "min_x": float(mins[0]), "min_y": float(mins[1]), "min_z": float(mins[2]),
        "max_x": float(maxs[0]), "max_y": float(maxs[1]), "max_z": float(maxs[2]),
        "centroid_x": float(centroid[0]), "centroid_y": float(centroid[1]), "centroid_z": float(centroid[2]),
        "volume": volume,
        "density": density
    }


def infer_class_from_processed_id(sample_id: str, manifest_map: Optional[Dict[str, str]] = None) -> str:
    """
    Infer class name from processed id (e.g., Buche_103 -> Buche) or from manifest_map if provided.
    manifest_map maps id -> raw_path; raw_path parent name is used as class.
    """
    if manifest_map and sample_id in manifest_map:
        raw_path = manifest_map[sample_id]
        try:
            cls = Path(raw_path).parent.name
            if cls:
                return cls
        except Exception:
            pass
    # fallback: split on underscore and take first token
    if "_" in sample_id:
        return sample_id.split("_")[0]
    # if id contains class prefix separated by hyphen or dot
    for sep in ("-", "."):
        if sep in sample_id:
            return sample_id.split(sep)[0]
    # final fallback: unknown
    return "unknown"


def build_manifest_map(manifest_path: Path) -> Dict[str, str]:
    """
    Read existing manifest.csv if present and build id -> raw_path mapping.
    """
    mapping: Dict[str, str] = {}
    if manifest_path.exists():
        try:
            df = pd.read_csv(manifest_path)
            if "id" in df.columns and "raw_path" in df.columns:
                for _, r in df.iterrows():
                    mapping[str(r["id"])] = str(r["raw_path"])
        except Exception as e:
            logger.warning(f"Failed to read manifest {manifest_path}: {e}")
    return mapping


def make_report(out_dir: Path,
                source: str,
                sample_table: pd.DataFrame,
                class_summary: pd.DataFrame,
                plots_paths: Dict[str, str],
                read_errors_path: Path,
                flagged_small_classes: List[str]) -> None:
    """
    Write Markdown report with tables and embedded relative links to plots.
    """
    ensure_dir(out_dir)
    report_path = out_dir / "eda_summary.md"
    ts = datetime.datetime.now().isoformat(sep=" ", timespec="seconds")
    lines: List[str] = []
    lines.append(f"# EDA Summary\n\nGenerated: {ts}\n")
    lines.append(f"Source used: {source}\n")
    lines.append(f"Total samples discovered: {len(sample_table)}\n")
    lines.append("## Per-class counts\n")
    lines.append(class_summary.to_markdown(index=False))
    lines.append("\n## Class balance plots\n")
    if plots_paths.get("class_balance_bar"):
        lines.append(f"![class_balance]({plots_paths['class_balance_bar']})\n")
    if plots_paths.get("point_count_histogram"):
        lines.append(f"![histogram]({plots_paths['point_count_histogram']})\n")
    if plots_paths.get("boxplot"):
        lines.append(f"![boxplot]({plots_paths['boxplot']})\n")
    lines.append("\n## Flagged small classes (< 30 samples)\n")
    if flagged_small_classes:
        for c in flagged_small_classes:
            lines.append(f"- {c}")
    else:
        lines.append("None")
    lines.append("\n## Data quality notes\n")
    issues = []
    if read_errors_path.exists():
        issues.append(f"See read errors log: {read_errors_path.name}")
    # very small point clouds (n_points < 10) - list up to 10
    tiny = sample_table[sample_table["n_points"] < 10]
    if not tiny.empty:
        issues.append(f"{len(tiny)} files with <10 points (examples: {', '.join(tiny['id'].astype(str).head(10).tolist())})")
    if issues:
        for it in issues:
            lines.append(f"- {it}")
    else:
        lines.append("- No major issues detected.")
    # write report
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(lines))
    logger.info(f"Markdown report written to {report_path}")


def create_plots(sample_table: pd.DataFrame, out_dir: Path, fast: bool = False) -> Dict[str, str]:
    """
    Create histogram, class balance bar, and boxplot. Return dict of relative paths.
    """
    plots_dir = out_dir / "plots"
    ensure_dir(plots_dir)
    plots = {}
    # Global histogram of point counts
    plt.figure(figsize=(6, 4) if not fast else (4, 3))
    plt.hist(sample_table["n_points"], bins=40 if not fast else 12, color="steelblue", edgecolor="k")
    plt.xlabel("Number of points")
    plt.ylabel("Files")
    plt.title("Point count per file (global)")
    hist_path = plots_dir / "point_count_histogram.png"
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()
    plots["point_count_histogram"] = os.path.relpath(hist_path, out_dir)

    # Class balance bar
    plt.figure(figsize=(8, 4) if not fast else (5, 3))
    class_counts = sample_table.groupby("class")["id"].count().sort_values(ascending=False)
    class_counts.plot(kind="bar", color="coral")
    plt.ylabel("Number of files")
    plt.title("Class balance (number of files per class)")
    cb_path = plots_dir / "class_balance_bar.png"
    plt.tight_layout()
    plt.savefig(cb_path)
    plt.close()
    plots["class_balance_bar"] = os.path.relpath(cb_path, out_dir)

    # Boxplot of points per class
    plt.figure(figsize=(10, 6) if not fast else (6, 4))
    sample_table.boxplot(column="n_points", by="class", rot=45)
    plt.suptitle("")
    plt.title("Points per file by class")
    plt.xlabel("Class")
    plt.ylabel("n_points")
    bp_path = plots_dir / "points_boxplot_per_class.png"
    plt.tight_layout()
    plt.savefig(bp_path)
    plt.close()
    plots["boxplot"] = os.path.relpath(bp_path, out_dir)

    logger.info(f"Plots saved to {plots_dir}")
    return plots


def run_eda(raw_dir: Path,
            processed_dir: Path,
            out_dir: Path,
            n_preview: int,
            max_samples: Optional[int],
            fast: bool,
            seed: int,
            verbose: bool):
    random.seed(seed)
    np.random.seed(seed)

    ensure_dir(out_dir)
    ensure_dir(out_dir / "sample_images")
    read_errors_path = out_dir / "read_errors.log"
    # reset read_errors.log
    if read_errors_path.exists():
        read_errors_path.unlink()

    mode = "processed" if processed_dir.exists() and any(processed_dir.rglob("*.npy")) else "raw"
    logger.info(f"EDA mode: {mode}")

    manifest_map = {}
    manifest_csv = processed_dir.parent / "manifest.csv" if processed_dir.parent else None
    if manifest_csv and manifest_csv.exists():
        manifest_map = build_manifest_map(manifest_csv)

    # gather files
    files: List[Path] = []
    sample_rows: List[Dict] = []

    if mode == "processed":
        # get .npy files
        files = sorted([p for p in processed_dir.rglob("*.npy") if p.is_file()])
        source_used = str(processed_dir)
    else:
        # raw mode: scan raw_dir for supported extensions
        files = []
        for ext in SUPPORTED_EXTS:
            files.extend(raw_dir.rglob(f"*{ext}"))
        files = sorted([p for p in files if p.is_file()])
        source_used = str(raw_dir)

    if max_samples is not None:
        files = files[:max_samples]
    if fast:
        files = files[:20]

    logger.info(f"Found {len(files)} candidate files to inspect")

    # limit processing memory: process sequentially with tqdm
    for p in tqdm(files, desc="Inspecting files"):
        try:
            # id and class inference
            if mode == "processed":
                sample_id = p.stem
                cls = infer_class_from_processed_id(sample_id, manifest_map)
                pts = load_pointcloud(p)
            else:
                sample_id = p.stem
                # class is parent folder name if present
                cls = p.parent.name if p.parent and p.parent != raw_dir else "unknown"
                pts = load_pointcloud(p)
            stats = compute_stats(pts)
            sample_rows.append({
                "id": sample_id,
                "class": cls,
                "n_points": stats["n_points"],
                "min_x": stats["min_x"],
                "min_y": stats["min_y"],
                "min_z": stats["min_z"],
                "max_x": stats["max_x"],
                "max_y": stats["max_y"],
                "max_z": stats["max_z"],
                "centroid_x": stats["centroid_x"],
                "centroid_y": stats["centroid_y"],
                "centroid_z": stats["centroid_z"],
                "volume": stats["volume"],
                "density": stats["density"],
                "path": str(p.resolve())
            })
        except Exception as e:
            logger.warning(f"Failed to read {p}: {e}")
            with open(read_errors_path, "a", encoding="utf-8") as fh:
                fh.write(f"{p}\t{type(e).__name__}\t{e}\n")
            continue

    if len(sample_rows) == 0:
        logger.error("No valid samples read; aborting EDA.")
        return {"mode": mode, "n_samples": 0, "report": None}

    df = pd.DataFrame(sample_rows)
    # Save point_counts_per_file.csv exactly
    ensure_dir(out_dir)
    pcpf = out_dir / "point_counts_per_file.csv"
    df.to_csv(pcpf, index=False)
    logger.info(f"Wrote {pcpf}")

    # Aggregation per class
    grp = df.groupby("class")["n_points"]
    class_summary = grp.agg(n_files="count", total_points="sum", mean_points="mean",
                            median_points="median", std_points="std").reset_index()
    # fill NaN stds with 0
    class_summary["std_points"] = class_summary["std_points"].fillna(0.0)
    # Save class_counts.csv exactly
    class_counts_csv = out_dir / "class_counts.csv"
    class_summary.to_csv(class_counts_csv, index=False)
    logger.info(f"Wrote {class_counts_csv}")

    # Flag small classes
    flagged = class_summary[class_summary["n_files"] < 30]["class"].tolist()

    # Create plots
    plots = create_plots(df, out_dir, fast=fast)

    # Create previews per class
    classes = sorted(df["class"].unique())
    rng = random.Random(seed)
    for cls in classes:
        cls_df = df[df["class"] == cls]
        n_avail = len(cls_df)
        if n_avail == 0:
            continue
        n_take = min(n_preview, n_avail)
        chosen = cls_df.sample(n=n_take, random_state=seed)
        for _, row in chosen.iterrows():
            try:
                pt_path = Path(row["path"])
                pts = load_pointcloud(pt_path)
                out_img = out_dir / "sample_images" / f"{cls}_{row['id']}_preview.png"
                if _HAS_OPEN3D:
                    # attempt Open3D render; fallback inside function to matplotlib
                    render_preview_open3d(pts, out_img)
                else:
                    sample_preview_views_matplotlib(pts, out_img)
            except Exception as e:
                logger.warning(f"Failed to create preview for {row['id']}: {e}")
                with open(read_errors_path, "a", encoding="utf-8") as fh:
                    fh.write(f"preview_failed\t{row['path']}\t{type(e).__name__}\t{e}\n")
                continue

    # Make markdown report
    make_report(out_dir, source_used, df, class_summary, plots, read_errors_path, flagged)

    return {"mode": mode, "n_samples": len(df), "report": str((out_dir / "eda_summary.md").resolve())}


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir)
    n_preview = int(args.n_preview)
    max_samples = int(args.max_samples) if args.max_samples is not None else None
    fast = bool(args.fast)
    seed = int(args.seed)
    verbose = bool(args.verbose)

    if verbose:
        logger.setLevel(logging.DEBUG)

    result = run_eda(raw_dir, processed_dir, out_dir, n_preview, max_samples, fast, seed, verbose)
    if result["n_samples"] == 0:
        logger.error("EDA completed with zero samples analyzed.")
        return 2
    logger.info(f"EDA finished. Mode: {result['mode']}. N_samples_analyzed: {result['n_samples']}. Report: {result['report']}")
    print(f"EDA finished. Mode: {result['mode']}. N_samples_analyzed: {result['n_samples']}. Report: {result['report']}")
    return 0


if __name__ == "__main__":
    # If no args provided, run a safe fast EDA and print short summary.
    if len(sys.argv) == 1:
        argv = ["--processed_dir", str(DEFAULT_PROCESSED_DIR), "--out_dir", "reports", "--fast", "--seed", "42"]
        # ensure deterministic
        exit_code = main(argv)
        sys.exit(exit_code)
    else:
        sys.exit(main(sys.argv[1:]))