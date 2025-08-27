#!/usr/bin/env python3
"""
render_views.py

Multi-view offscreen renderer for sampled point clouds.

Usage examples:

Debug / fast run (use processed samples):
    python render_views.py --sampled_dir data/processed/sampled --out_dir data/processed/views --fast --workers 1 --verbose

Full run (use multiple workers, produce previews):
    python render_views.py --sampled_dir ../../data/processed/sampled --out_dir ../../data/processed/views/ --views 12 --resolution 224 --workers 4 --n_preview 20
    

This script attempts to use Open3D or pyrender+trimesh for offscreen rendering. If neither
is available it falls back to a deterministic matplotlib-based 2D projection renderer.
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path
import math
import datetime
import yaml
import multiprocessing as mp
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageOps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional backends
try:
    import open3d as o3d  # type: ignore
    _HAS_OPEN3D = True
except Exception:
    o3d = None
    _HAS_OPEN3D = False

try:
    import trimesh  # type: ignore
    import pyrender  # type: ignore
    _HAS_PYRENDER = True
except Exception:
    trimesh = None
    pyrender = None
    _HAS_PYRENDER = False

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("render_views")


# -----------------------
# Utility & IO functions
# -----------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_sample(path: Path) -> np.ndarray:
    """
    Load a sampled point cloud from .npy file and return an (N,3) float64 numpy array.
    Raises exception if invalid.
    """
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"invalid shape {arr.shape} (expected Nx3) for {path}")
    return arr[:, :3].astype(np.float64)


def bbox_and_center(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute bbox min, max, centroid and diagonal length.
    Returns (min_xyz, max_xyz, diagonal_length)
    """
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centroid = points.mean(axis=0)
    diag = float(np.linalg.norm(maxs - mins))
    return mins, maxs, centroid, diag


# -----------------------
# Camera pose utilities
# -----------------------
def compute_camera_poses(n_views: int,
                         bbox_min: np.ndarray,
                         bbox_max: np.ndarray,
                         centroid: np.ndarray,
                         radius_factor: float = 2.0,
                         seed: int = 42,
                         elevation_levels: Optional[List[float]] = None
                         ) -> List[Dict[str, np.ndarray]]:
    """
    Compute list of camera poses (position, lookat, up) deterministically.

    - elevation_levels: list of elevations in degrees (e.g., [-30, 0, 30]).
      If None and n_views is divisible by 3, default to [-30, 0, 30].
      Otherwise compute a reasonable grid.
    - Azimuths are spaced uniformly per elevation.

    Returns: list of dicts with keys: position (3,), lookat (3,), up (3,)
    """
    if elevation_levels is None:
        # default 3 levels if possible
        if n_views % 3 == 0:
            elevation_levels = [-30.0, 0.0, 30.0]
        else:
            # try 3 levels anyway
            elevation_levels = [-30.0, 0.0, 30.0]

    n_elev = len(elevation_levels)
    # determine azimuths per elevation
    azimuths = []
    # distribute views across elevation levels approximately evenly
    per_level = max(1, n_views // n_elev)
    remaining = n_views - per_level * n_elev
    counts = [per_level] * n_elev
    for i in range(remaining):
        counts[i] += 1

    # compute bbox diagonal -> radius
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    radius = diag * radius_factor if diag > 1e-9 else radius_factor

    poses = []
    rng = np.random.RandomState(seed)
    for lvl_idx, elev in enumerate(elevation_levels):
        cnt = counts[lvl_idx]
        # uniform azimuths for this level
        for k in range(cnt):
            az = float((360.0 * k) / max(1, cnt))
            # add tiny deterministic jitter from RNG to avoid exact degeneracy
            jitter = float(rng.uniform(-1e-6, 1e-6))
            az += jitter
            # convert to radians
            el_rad = math.radians(elev)
            az_rad = math.radians(az)
            x = radius * math.cos(el_rad) * math.cos(az_rad)
            y = radius * math.cos(el_rad) * math.sin(az_rad)
            z = radius * math.sin(el_rad)
            position = np.array([x, y, z], dtype=np.float64) + centroid
            lookat = centroid.copy()
            up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            # ensure up not parallel to view direction
            fwd = (lookat - position)
            if np.linalg.norm(fwd) < 1e-9:
                fwd = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            fwd = fwd / np.linalg.norm(fwd)
            if abs(np.dot(fwd, up)) > 0.999:
                up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            poses.append({"position": position, "lookat": lookat, "up": up, "elevation": elev, "azimuth": az})
    # trim or pad poses to exactly n_views
    if len(poses) > n_views:
        poses = poses[:n_views]
    elif len(poses) < n_views:
        # replicate last poses deterministically
        while len(poses) < n_views:
            poses.append(poses[len(poses) % len(poses)])
    return poses


# -----------------------
# Rendering backends
# -----------------------
def render_offscreen_open3d(points: np.ndarray,
                            camera_pose: Dict[str, np.ndarray],
                            resolution: int,
                            point_size: float,
                            bg_color: str,
                            ortho: bool) -> Optional[Image.Image]:
    """
    Render points with Open3D offscreen renderer. Returns PIL Image or None on failure.
    """
    if not _HAS_OPEN3D:
        return None
    try:
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        # colors: dark gray
        colors = np.tile(np.array([[0.15, 0.2, 0.15]], dtype=np.float64), (points.shape[0], 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Setup renderer
        # OffscreenRenderer requires Open3D built with rendering support.
        w = resolution
        h = resolution
        renderer = o3d.visualization.rendering.OffscreenRenderer(w, h)
        scene = renderer.scene
        scene.set_background([1.0, 1.0, 1.0, 0.0] if bg_color == "white" else [0.0, 0.0, 0.0, 0.0])
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"  # simple shader
        scene.add_geometry("pcd", pcd, material)

        center = camera_pose["lookat"]
        eye = camera_pose["position"]
        up = camera_pose["up"]
        # Field of view and projection
        if ortho:
            # approximate orthographic by using very large focal length / parallel projection is not exposed easily
            # Fall back to setting a narrow fov
            fov = 1.0
        else:
            fov = 60.0

        # setup camera
        cam = renderer.scene.camera
        cam.setup_camera(fov, center, eye, up)
        # point size via material/geometry is limited; try to set render options
        # Render to image
        img = renderer.render_to_image()
        # convert to PIL
        arr = np.asarray(img)
        pil = Image.fromarray(arr)
        renderer.release()
        return pil
    except Exception as e:
        logger.debug(f"Open3D render failed: {e}")
        return None


def render_offscreen_pyrender(points: np.ndarray,
                              camera_pose: Dict[str, np.ndarray],
                              resolution: int,
                              point_size: float,
                              bg_color: str,
                              ortho: bool) -> Optional[Image.Image]:
    """
    Render points with pyrender + trimesh offscreen. Returns PIL Image or None on failure.
    """
    if not _HAS_PYRENDER:
        return None
    try:
        # Convert to trimesh PointCloud via vertices and small spheres approximated by points (point primitives unsupported)
        mesh = trimesh.PointCloud(points.astype(np.float64))
        scene = pyrender.Scene(bg_color=(255, 255, 255, 0) if bg_color == "white" else (0, 0, 0, 0), ambient_light=(0.8, 0.8, 0.8))
        # create mesh node
        try:
            node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False))
            scene.add_node(node)
        except Exception:
            # fallback: add points as small spheres is expensive; instead convert to a mesh via voxelization? skip
            pass

        # camera
        center = camera_pose["lookat"]
        eye = camera_pose["position"]
        up = camera_pose["up"]
        # compute camera transform
        camera = pyrender.PerspectiveCamera(yfov=math.radians(60.0))
        cam_pose = lookat_matrix(eye, center, up)
        scene.add(camera, pose=cam_pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        scene.add(light, pose=cam_pose)
        r = pyrender.OffscreenRenderer(viewport_width=resolution, viewport_height=resolution)
        color, _ = r.render(scene)
        r.delete()
        img = Image.fromarray(color)
        return img
    except Exception as e:
        logger.debug(f"pyrender render failed: {e}")
        return None


def lookat_matrix(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Return 4x4 camera-to-world matrix for pyrender given eye, center, up.
    """
    # Based on gluLookAt
    f = (center - eye)
    f = f / (np.linalg.norm(f) + 1e-12)
    upn = up / (np.linalg.norm(up) + 1e-12)
    s = np.cross(f, upn)
    s = s / (np.linalg.norm(s) + 1e-12)
    u = np.cross(s, f)
    mat = np.eye(4, dtype=np.float64)
    mat[0, :3] = s
    mat[1, :3] = u
    mat[2, :3] = -f
    mat[:3, 3] = eye
    return mat


def render_fallback_matplotlib(points: np.ndarray,
                               camera_pose: Dict[str, np.ndarray],
                               resolution: int,
                               point_size: float,
                               bg_color: str,
                               ortho: bool) -> Image.Image:
    """
    Render a single view using matplotlib 2D projection into a PIL Image.
    Projects points onto camera plane defined by camera_pose (position, lookat, up).
    """
    # compute camera basis
    eye = camera_pose["position"]
    center = camera_pose["lookat"]
    up = camera_pose["up"].astype(np.float64)

    forward = (center - eye)
    if np.linalg.norm(forward) < 1e-12:
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-9:
        # choose alternative up
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-12)
    true_up = np.cross(right, forward)
    true_up = true_up / (np.linalg.norm(true_up) + 1e-12)

    # compute coordinates in camera plane
    rel = points - center  # center-based coordinates for better centering
    x_coords = rel.dot(right)
    y_coords = rel.dot(true_up)
    z_coords = rel.dot(forward)  # depth

    # scaling for visualization: use bbox diagonal of points to set scale
    bbox_diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    if bbox_diag < 1e-9:
        bbox_diag = 1.0
    scale = (resolution * 0.35) / (bbox_diag)  # empirical
    xs = x_coords * scale
    ys = y_coords * scale

    # prepare figure
    fig = plt.figure(figsize=(resolution / 100.0, resolution / 100.0), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_facecolor("white" if bg_color == "white" else "black")
    # plot points, use small size derived from point_size
    s = max(0.1, float(point_size))  # marker size in matplotlib points^2
    ax.scatter(xs, ys, s=s, color=(0.0, 0.0, 0.0) if bg_color == "white" else (1.0, 1.0, 1.0), marker="o", linewidths=0, alpha=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(np.min(xs) - 1, np.max(xs) + 1)
    ax.set_ylim(np.min(ys) - 1, np.max(ys) + 1)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout(pad=0)
    # save to buffer
    buf = None
    try:
        from io import BytesIO
        buf = BytesIO()
        fig.canvas.print_png(buf)
        buf.seek(0)
        img = Image.open(buf).convert("RGBA" if bg_color != "white" else "RGB")
    finally:
        plt.close(fig)
        if buf:
            buf.close()
    # if background requested transparent, set alpha accordingly
    return img.convert("RGBA") if bg_color != "white" else img.convert("RGB")


def save_image(img: Image.Image, path: Path, transparent: bool) -> None:
    ensure_dir(path.parent)
    try:
        if transparent and img.mode != "RGBA":
            img = img.convert("RGBA")
            # set white background as transparent if originally white
            datas = img.getdata()
            newData = []
            for item in datas:
                if item[0] > 250 and item[1] > 250 and item[2] > 250:
                    newData.append((255, 255, 255, 0))
                else:
                    newData.append(item)
            img.putdata(newData)
        img.save(str(path), format="PNG")
    except Exception as e:
        # fallback to simple save
        img.convert("RGB").save(str(path), format="PNG")


def save_collage(img_paths: List[Path], out_preview_path: Path, thumbnail_w: int = 224, bg_color: str = "white") -> None:
    """
    Create a horizontal collage of up to 3 images (or more if provided) and save to out_preview_path.
    """
    if not img_paths:
        return
    thumbs = []
    for p in img_paths[:3]:
        try:
            im = Image.open(p)
            im = ImageOps.contain(im, (thumbnail_w, thumbnail_w))
            thumbs.append(im.convert("RGBA") if im.mode == "RGBA" else im.convert("RGB"))
        except Exception:
            # placeholder
            placeholder = Image.new("RGB", (thumbnail_w, thumbnail_w), color=(255, 255, 255) if bg_color == "white" else (0, 0, 0))
            thumbs.append(placeholder)
    # compose horizontally
    total_w = sum(im.width for im in thumbs)
    h = max(im.height for im in thumbs)
    collage = Image.new("RGB", (total_w, h), color=(255, 255, 255) if bg_color == "white" else (0, 0, 0))
    x = 0
    for im in thumbs:
        collage.paste(im.convert("RGB"), (x, 0))
        x += im.width
    ensure_dir(out_preview_path.parent)
    collage.save(str(out_preview_path), format="PNG")


# -----------------------
# Per-sample rendering orchestration
# -----------------------
def render_sample_worker(args_tuple: Tuple) -> Dict[str, Any]:
    """
    Worker function to render one sample. Returns a record dict for manifest or error info.
    args_tuple contains:
      (sample_path_str, out_dir_str, resolution, views, point_size, bg_color, transparent, ortho,
       camera_radius_factor, seed, overwrite, reports_dir_str)
    """
    (sample_path_str, out_dir_str, resolution, views, point_size, bg_color, transparent, ortho,
     camera_radius_factor, seed, overwrite, reports_dir_str) = args_tuple

    sample_path = Path(sample_path_str)
    out_dir = Path(out_dir_str)
    reports_dir = Path(reports_dir_str)
    rec = {
        "id": sample_path.stem,
        "sample_path": str(sample_path.resolve()),
        "views_dir": "",
        "n_views": 0,
        "note": "",
        "rendered_timestamp": ""
    }

    try:
        pts = load_sample(sample_path)
        if pts.shape[0] < 1:
            raise ValueError("empty point cloud")

        mins, maxs, centroid, diag = bbox_and_center(pts)
        # camera poses
        poses = compute_camera_poses(views, mins, maxs, centroid, radius_factor=camera_radius_factor, seed=seed)
        sample_views_dir = out_dir / sample_path.stem
        ensure_dir(sample_views_dir)

        saved_paths = []
        for idx, pose in enumerate(poses):
            view_idx = idx + 1
            fname = f"view_{view_idx:02d}.png"
            out_path = sample_views_dir / fname
            # skip if exists and not overwrite
            if out_path.exists() and not overwrite:
                saved_paths.append(out_path)
                continue

            img = None
            # try backends in order
            if _HAS_OPEN3D:
                try:
                    img = render_offscreen_open3d(pts, pose, resolution, point_size, bg_color, ortho)
                except Exception:
                    img = None
            if img is None and _HAS_PYRENDER:
                try:
                    img = render_offscreen_pyrender(pts, pose, resolution, point_size, bg_color, ortho)
                except Exception:
                    img = None
            if img is None:
                # fallback to matplotlib projection
                img = render_fallback_matplotlib(pts, pose, resolution, point_size, bg_color, ortho)

            save_image(img, out_path, transparent)
            saved_paths.append(out_path)

        rec["views_dir"] = str(sample_views_dir.resolve())
        rec["n_views"] = len(saved_paths)
        rec["note"] = "ok"
        rec["rendered_timestamp"] = datetime.datetime.now().isoformat(sep=" ", timespec="seconds")
        return {"success": True, "record": rec, "saved_paths": [str(p) for p in saved_paths]}
    except Exception as e:
        # write error log
        try:
            ensure_dir(reports_dir)
            with open(reports_dir / "read_errors.log", "a", encoding="utf-8") as fh:
                fh.write(f"{sample_path}\t{type(e).__name__}\t{e}\n")
        except Exception:
            pass
        logger.debug(f"Failed to render {sample_path}: {e}")
        rec["note"] = f"failed: {e}"
        rec["rendered_timestamp"] = datetime.datetime.now().isoformat(sep=" ", timespec="seconds")
        return {"success": False, "record": rec, "saved_paths": []}


# Worker initializer at module level for multiprocessing picklability
def _init_worker(seed: int):
    """
    Initialize RNG for a worker. Must be module-level for multiprocessing on Windows.
    """
    try:
        from multiprocessing import current_process
        ident = getattr(current_process(), "_identity", None)
        wid = ident[0] if ident else 0
    except Exception:
        wid = 0
    np.random.seed((int(seed) + int(wid)) % (2 ** 31 - 1))


# -----------------------
# Main orchestration
# -----------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Render multi-view images for sampled point clouds.")
    p.add_argument("--sampled_dir", type=str, default="../../data/processed/sampled", help="Directory with sampled .npy files")
    p.add_argument("--raw_dir", type=str, default="", help="Optional raw point cloud dir to prefer if matching")
    p.add_argument("--out_dir", type=str, default="../../data/processed/views/", help="Output directory for views")
    p.add_argument("--views", type=int, default=12, help="Number of views to render (default 12)")
    p.add_argument("--resolution", type=int, default=224, help="Square output resolution (default 224)")
    p.add_argument("--point_size", type=float, default=2.0, help="Point size for rendering (default 2.0)")
    p.add_argument("--bg_color", type=str, default="white", choices=["white", "black"], help="Background color")
    p.add_argument("--transparent", action="store_true", help="Save PNG with transparent background where supported")
    p.add_argument("--ortho", action="store_true", help="Use orthographic projection if backend supports it")
    p.add_argument("--n_preview", type=int, default=10, help="Number of sample preview collages to create")
    p.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    p.add_argument("--fast", action="store_true", help="Fast mode: limit to 20 samples and n_preview=10")
    p.add_argument("--max_samples", type=int, default=None, help="Limit number of samples processed")
    p.add_argument("--camera_radius_factor", type=float, default=2.0, help="Camera radius factor relative to bbox diag")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing images")
    p.add_argument("--seed", type=int, default=42, help="Random seed for determinism")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    sampled_dir = Path(args.sampled_dir)
    raw_dir = Path(args.raw_dir) if args.raw_dir else None
    out_dir = Path(args.out_dir)
    reports_dir = Path("../../reports")
    ensure_dir(out_dir)
    ensure_dir(reports_dir)
    ensure_dir(reports_dir / "sample_images")

    if args.fast:
        max_samples = 20
        n_preview = min(args.n_preview, 10)
    else:
        max_samples = args.max_samples
        n_preview = args.n_preview

    # discover sampled .npy files
    files = sorted([p for p in sampled_dir.rglob("*.npy") if p.is_file()], key=lambda p: str(p))
    if max_samples is not None:
        files = files[:max_samples]
    total = len(files)
    logger.info(f"Found {total} sampled .npy files under {sampled_dir}.")

    if total == 0:
        logger.warning("No files to process; exiting.")
        return 0

    # Prepare tasks
    tasks = []
    for p in files:
        tasks.append((str(p), str(out_dir), int(args.resolution), int(args.views),
                      float(args.point_size), args.bg_color, bool(args.transparent), bool(args.ortho),
                      float(args.camera_radius_factor), int(args.seed), bool(args.overwrite), str(reports_dir)))

    results = []
    if args.workers > 1:
        try:
            with mp.Pool(processes=args.workers, initializer=_init_worker, initargs=(int(args.seed),)) as pool:
                for res in tqdm(pool.imap_unordered(render_sample_worker, tasks), total=len(tasks), desc="Rendering"):
                    results.append(res)
        except Exception as e:
            logger.error(f"Multiprocessing failed: {e}. Falling back to single-process.")
            for t in tqdm(tasks, desc="Rendering (fallback)"):
                results.append(render_sample_worker(t))
    else:
        for t in tqdm(tasks, desc="Rendering"):
            results.append(render_sample_worker(t))

    # Collect manifest records and write manifest_views.csv
    manifest_records = []
    saved_preview_ids = []
    for r in results:
        rec = r.get("record", {})
        manifest_records.append(rec)
    manifest_df = pd.DataFrame(manifest_records, columns=["id", "sample_path", "views_dir", "n_views", "note", "rendered_timestamp"])

    manifest_path = out_dir / "manifest_views.csv"
    # If overwrite flag, replace; else append new entries but avoid duplicates (keep newest)
    if manifest_path.exists() and not args.overwrite:
        try:
            existing = pd.read_csv(manifest_path)
            # concat and drop duplicates by id, keeping last
            combined = pd.concat([existing, manifest_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["id"], keep="last")
            combined.to_csv(manifest_path, index=False)
        except Exception:
            manifest_df.to_csv(manifest_path, index=False)
    else:
        manifest_df.to_csv(manifest_path, index=False)
    logger.info(f"Wrote manifest to {manifest_path}")

    # Save render configuration for reproducibility
    config = {
        "timestamp": datetime.datetime.now().isoformat(sep=" ", timespec="seconds"),
        "seed": int(args.seed),
        "views": int(args.views),
        "resolution": int(args.resolution),
        "point_size": float(args.point_size),
        "bg_color": args.bg_color,
        "transparent": bool(args.transparent),
        "ortho": bool(args.ortho),
        "n_preview": int(n_preview),
        "workers": int(args.workers),
        "camera_radius_factor": float(args.camera_radius_factor),
        "overwrite": bool(args.overwrite)
    }
    render_config_path = out_dir / "render_config.yaml"
    try:
        with open(render_config_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(config, fh)
        logger.info(f"Wrote render config to {render_config_path}")
    except Exception:
        logger.debug("Failed to write render_config.yaml")

    # Create previews for first n_preview successful samples
    success_rows = manifest_df[manifest_df["note"] == "ok"]
    preview_rows = success_rows.head(n_preview)
    for _, row in preview_rows.iterrows():
        vid = row["id"]
        views_dir = Path(row["views_dir"])
        # choose representative view files: view_01, view_05, view_09 if present, else first 3
        chosen = []
        for vname in ["view_01.png", "view_05.png", "view_09.png"]:
            p = views_dir / vname
            if p.exists():
                chosen.append(p)
        if len(chosen) < 3:
            # take first three pngs in dir
            pngs = sorted([p for p in views_dir.glob("*.png")])
            chosen = pngs[:3]
        preview_out = reports_dir / "sample_images" / f"{vid}_preview.png"
        save_collage(chosen, preview_out, thumbnail_w=args.resolution, bg_color=args.bg_color)
        saved_preview_ids.append(vid)

    mode = "fast" if args.fast else "full"
    n_processed = len(manifest_df)
    print(f"Render finished. Mode: {mode}. Samples processed: {n_processed}. Views saved to {out_dir}.")
    # sanity check: print first manifest row if exists
    try:
        if manifest_path.exists():
            dfm = pd.read_csv(manifest_path, nrows=1)
            print("First manifest row preview:")
            print(dfm.to_dict(orient="records")[0])
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    # safe debug when no args
    if len(sys.argv) == 1:
        argv = ["--sampled_dir", "../../data/processed/sampled", "--out_dir", "../../data/processed/views", "--fast", "--workers", "1"]
        sys.exit(main(argv))
    else:
        sys.exit(main(sys.argv[1:]))   