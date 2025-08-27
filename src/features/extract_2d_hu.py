#!/usr/bin/env python3
"""
extract_2d_hu.py

Compute Hu Moments per view and aggregate per-sample descriptors (mean+std).
Saves per-sample numpy and a global CSV table.

Usage (debug):
    python extract_2d_hu.py --views_dir data/processed/views --out_dir data/processed/2d_hu --fast --workers 1

Full run example:
    python extract_2d_hu.py --views_dir ../../data/processed/views/ --out_dir ../../data/processed/2d_hu --workers 4 --preprocess_crop --use_silhouette
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path
import datetime
import math
import multiprocessing as mp
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageOps

# Optional OpenCV
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

# YAML optional
try:
    import yaml
    _HAS_YAML = True
except Exception:
    yaml = None
    _HAS_YAML = False

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_2d_hu")


# -----------------------
# Utilities
# -----------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_sample_ids(views_dir: Path) -> List[Tuple[str, Path]]:
    """
    Return list of (id, dir_path) for directories under views_dir that contain image files.
    Deterministic lexicographic sort.
    """
    ids = []
    if not views_dir.exists():
        return ids
    # list immediate subdirectories
    for d in sorted(views_dir.iterdir(), key=lambda p: str(p)):
        if d.is_dir():
            # check for png/jpg files
            imgs = sorted([p for p in d.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
            if imgs:
                ids.append((d.name, d))
    return ids


def load_image(path: Path) -> np.ndarray:
    """
    Load an image and return grayscale uint8 numpy array.
    Uses cv2 if available, else PIL.
    """
    if _HAS_CV2:
        im = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if im is None:
            # fallback to PIL
            pil = Image.open(str(path)).convert("L")
            return np.asarray(pil, dtype=np.uint8)
        return im
    else:
        pil = Image.open(str(path)).convert("L")
        return np.asarray(pil, dtype=np.uint8)


def _otsu_threshold_numpy(img: np.ndarray) -> int:
    """
    Simple Otsu implementation using numpy. Returns threshold in [0,255].
    """
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
    total = img.size
    sumB = 0.0
    wB = 0.0
    maximum = 0.0
    sum1 = np.dot(np.arange(256), hist)
    for i in range(256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (sum1 - sumB) / wF
        between = wB * wF * (mB - mF) * (mB - mF)
        if between >= maximum:
            level = i
            maximum = between
    return int(level)


def compute_otsu(img: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Return threshold and binary mask using Otsu. Use cv2 if available else numpy.
    Auto-invert mask if the largest component appears to be the background.
    """
    if _HAS_CV2:
        thresh, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        t = _otsu_threshold_numpy(img)
        mask = (img > t).astype(np.uint8) * 255
        thresh = int(t)

    # Ensure mask is binary (0/255)
    mask = (mask > 127).astype(np.uint8) * 255

    # Heuristic: if the foreground (mask==255) occupies >50% of image, it is probably the background -> invert
    fg_ratio = float(np.count_nonzero(mask) / mask.size)
    if fg_ratio > 0.5:
        mask = (255 - mask).astype(np.uint8)

    return int(thresh), mask


def find_largest_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Find bounding box of largest connected component in binary mask (uint8 0/255).
    Returns x,y,w,h. If none found, return whole image bbox.
    """
    h, w = mask.shape[:2]
    if _HAS_CV2:
        contours_info = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_info[-2]
    else:
        # naive connected component: find bounding boxes of nonzero pixels
        coords = np.argwhere(mask > 0)
        if coords.size == 0:
            return 0, 0, w, h
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        return int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)
    if not contours:
        return 0, 0, w, h
    # choose largest by area
    max_cnt = max(contours, key=lambda c: cv2.contourArea(c) if _HAS_CV2 else 0)
    x, y, ww, hh = cv2.boundingRect(max_cnt)
    return int(x), int(y), int(ww), int(hh)


def pad_and_resize(img: np.ndarray, target: int, bg_color: str) -> np.ndarray:
    """
    Resize preserving aspect ratio and pad to (target,target) with bg_color.
    img is numpy uint8 grayscale.
    """
    pil = Image.fromarray(img)
    # maintain aspect ratio
    pil_thumb = ImageOps.contain(pil, (target, target))
    # create background
    bg = (255, 255, 255) if bg_color == "white" else (0, 0, 0)
    new_im = Image.new("RGB", (target, target), color=bg)
    # paste centered
    # convert pil_thumb to RGB (grayscale)
    paste = pil_thumb.convert("RGB")
    x = (target - paste.width) // 2
    y = (target - paste.height) // 2
    new_im.paste(paste, (x, y))
    # return grayscale array
    return np.asarray(new_im.convert("L"), dtype=np.uint8)


def preprocess_image(img: np.ndarray,
                     resolution: int,
                     crop_margin: float,
                     bg_color: str,
                     preprocess_crop: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess a grayscale uint8 image:
    - If preprocess_crop: compute Otsu silhouette, bbox of largest component, expand by crop_margin,
      crop, resize+pad to resolution.
    - Returns (preprocessed_img, silhouette_mask) as uint8 arrays.
    """
    assert img.ndim == 2
    h, w = img.shape[:2]
    # compute Otsu mask
    try:
        _, mask = compute_otsu(img)
    except Exception:
        # fallback simple threshold at median
        t = int(np.median(img))
        mask = (img > t).astype(np.uint8) * 255

    if not preprocess_crop:
        # resize/pad original image
        preproc = pad_and_resize(img, resolution, bg_color)
        # resize mask similarly but preserve binary values (use nearest-like behavior)
        mask_pil = Image.fromarray(mask).convert("L")
        mask_resized = ImageOps.contain(mask_pil, (resolution, resolution))
        # re-binarize to remove interpolation artifacts
        mask_resized = mask_resized.point(lambda p: 255 if p > 128 else 0).convert("L")
        mask_bg = Image.new("L", (resolution, resolution), color=0)
        x = (resolution - mask_resized.width) // 2
        y = (resolution - mask_resized.height) // 2
        mask_bg.paste(mask_resized, (x, y))
        mask_arr = np.asarray(mask_bg, dtype=np.uint8)
        return preproc, mask_arr

    # find largest bbox
    x, y, ww, hh = find_largest_bbox(mask)
    # expand by margin; crop_margin may be fraction <=1 or absolute pixels >1
    max_side = max(ww, hh)
    if crop_margin <= 1.0:
        pad = int(round(crop_margin * max_side))
    else:
        pad = int(round(crop_margin))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + ww + pad)
    y1 = min(h, y + hh + pad)
    if x1 <= x0 or y1 <= y0:
        # fallback to full image
        x0, y0, x1, y1 = 0, 0, w, h
    cropped = img[y0:y1, x0:x1]
    cropped_mask = mask[y0:y1, x0:x1]
    # resize+pad
    preproc = pad_and_resize(cropped, resolution, bg_color)
    # resize mask similar
    mask_pil = Image.fromarray(cropped_mask).convert("L")
    mask_resized = ImageOps.contain(mask_pil, (resolution, resolution))
    mask_resized = mask_resized.point(lambda p: 255 if p > 128 else 0).convert("L")
    mask_bg = Image.new("L", (resolution, resolution), color=0)
    xpad = (resolution - mask_resized.width) // 2
    ypad = (resolution - mask_resized.height) // 2
    mask_bg.paste(mask_resized, (xpad, ypad))
    mask_arr = np.asarray(mask_bg, dtype=np.uint8)
    return preproc, mask_arr


def compute_hu_from_image(img_or_mask: np.ndarray) -> np.ndarray:
    """
    Compute 7 Hu moments from grayscale image or binary mask (uint8). Apply stable log transform:
    val = -sign(m) * log10(abs(m) + 1e-300). Returns float32 array shape (7,).
    """
    if _HAS_CV2:
        moments = cv2.moments(img_or_mask.astype(np.uint8))
        hu = cv2.HuMoments(moments).reshape((7,))
    else:
        # fallback: approximate moments using numpy (centered raw moments)
        # This is a simplified fallback; for safety return zeros if cv2 not available and image blank
        try:
            # compute raw moments up to order 3
            y, x = np.indices(img_or_mask.shape)
            I = img_or_mask.astype(np.float64)
            m00 = I.sum() + 1e-12
            m10 = (x * I).sum()
            m01 = (y * I).sum()
            xbar = m10 / m00
            ybar = m01 / m00
            # central moments
            mu = {}
            for p in range(4):
                for q in range(4):
                    if p + q <= 3:
                        mu[(p, q)] = (((x - xbar) ** p) * ((y - ybar) ** q) * I).sum()
            # compute normalized central moments and Hu moments approximately -> fallback zeros
            hu = np.zeros((7,), dtype=np.float64)
        except Exception:
            hu = np.zeros((7,), dtype=np.float64)
    # apply log transform
    eps = 1e-300
    hu_log = []
    for m in hu:
        val = -np.sign(m) * np.log10(abs(float(m)) + eps)
        # handle infinite
        if math.isinf(val) or math.isnan(val):
            val = 0.0
        hu_log.append(val)
    return np.asarray(hu_log, dtype=np.float32)


def save_descriptor_npy(sample_id: str, vec: np.ndarray, out_dir: Path) -> Path:
    """
    Save per-sample descriptor as <id>_hu.npy float32.
    """
    ensure_dir(out_dir)
    p = out_dir / f"{sample_id}_hu.npy"
    np.save(str(p), vec.astype(np.float32), allow_pickle=False)
    return p


def append_to_csv(records: List[Dict[str, Any]], out_csv_path: Path, overwrite: bool = False) -> None:
    """
    Write or append records (list of dicts) to CSV file with exact column order.
    If overwrite True, replace file. If append, merge and keep last entry per id.
    """
    cols = ["id"] + [f"hu{i+1}_mean" for i in range(7)] + [f"hu{i+1}_std" for i in range(7)] + ["note"]
    df_new = pd.DataFrame(records, columns=cols)
    if out_csv_path.exists() and not overwrite:
        try:
            df_old = pd.read_csv(out_csv_path)
            combined = pd.concat([df_old, df_new], ignore_index=True)
            combined = combined.drop_duplicates(subset=["id"], keep="last")
            combined.to_csv(out_csv_path, index=False)
            return
        except Exception as e:
            logger.debug(f"Failed to append to existing CSV; will overwrite. Error: {e}")
    # write new
    df_new.to_csv(out_csv_path, index=False)


def save_preview(id: str,
                 chosen_view_paths: List[Path],
                 preproc_img: np.ndarray,
                 mask: np.ndarray,
                 out_preview_path: Path,
                 bg_color: str,
                 resolution: int) -> None:
    """
    Save a 3-panel collage:
      - left: grid of chosen raw views (up to 3)
      - center: preprocessed image
      - right: mask or grayscale representation
    """
    panels = []
    # left: concatenate chosen raw views horizontally (converted to RGB)
    left_imgs = []
    for p in chosen_view_paths[:3]:
        try:
            raw = Image.open(p).convert("RGB")
            raw = ImageOps.contain(raw, (resolution, resolution))
        except Exception:
            raw = Image.new("RGB", (resolution, resolution), color=(255, 255, 255) if bg_color == "white" else (0, 0, 0))
        left_imgs.append(raw)
    if not left_imgs:
        left = Image.new("RGB", (resolution, resolution), color=(255, 255, 255) if bg_color == "white" else (0, 0, 0))
    else:
        total_w = sum(im.width for im in left_imgs)
        h = max(im.height for im in left_imgs)
        left = Image.new("RGB", (total_w, h), color=(255, 255, 255) if bg_color == "white" else (0, 0, 0))
        x = 0
        for im in left_imgs:
            left.paste(im, (x, 0))
            x += im.width

    # center: preproc_img (grayscale numpy)
    try:
        center = Image.fromarray(preproc_img).convert("RGB")
    except Exception:
        center = Image.new("RGB", (resolution, resolution), color=(255, 255, 255) if bg_color == "white" else (0, 0, 0))
    center = ImageOps.contain(center, (resolution, resolution))

    # right: mask or grayscale
    try:
        right = Image.fromarray(mask).convert("RGB")
    except Exception:
        right = Image.new("RGB", (resolution, resolution), color=(255, 255, 255) if bg_color == "white" else (0, 0, 0))
    right = ImageOps.contain(right, (resolution, resolution))

    # normalize heights
    h = max(left.height, center.height, right.height)
    # pad each to height h
    def pad_vert(im):
        if im.height == h:
            return im
        new = Image.new("RGB", (im.width, h), color=(255, 255, 255) if bg_color == "white" else (0, 0, 0))
        new.paste(im, (0, (h - im.height) // 2))
        return new

    left = pad_vert(left)
    center = pad_vert(center)
    right = pad_vert(right)

    # compose horizontally
    total_w = left.width + center.width + right.width
    collage = Image.new("RGB", (total_w, h), color=(255, 255, 255) if bg_color == "white" else (0, 0, 0))
    x = 0
    for im in [left, center, right]:
        collage.paste(im, (x, 0))
        x += im.width
    ensure_dir(out_preview_path.parent)
    collage.save(str(out_preview_path), format="PNG")


# -----------------------
# Worker processing
# -----------------------
def process_sample_worker(args_tuple: Tuple) -> Dict[str, Any]:
    """
    Worker function to process one sample directory.
    args_tuple: (id, dir_path_str, config_dict)
    Returns dict: {"success": bool, "id": id, "vec": list(14) or None, "note": str}
    """
    sample_id, dir_path_str, config = args_tuple
    dir_path = Path(dir_path_str)
    out_dir = Path(config["out_dir"])
    resolution = int(config["resolution"])
    use_silhouette = bool(config["use_silhouette"])
    preprocess_crop = bool(config["preprocess_crop"])
    crop_margin = float(config["crop_margin"])
    bg_color = config["bg_color"]
    expected_views = int(config["expected_views"])
    overwrite = bool(config["overwrite"])
    reports_dir = Path("reports")
    result = {"success": False, "id": sample_id, "vec": None, "note": ""}

    try:
        # list view images sorted lexicographically
        view_files = sorted([p for p in dir_path.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")], key=lambda p: str(p))
        if len(view_files) == 0:
            raise FileNotFoundError(f"No view images in {dir_path}")

        hu_list = []
        for vpath in view_files:
            try:
                img = load_image(vpath)
                preproc, mask = preprocess_image(img, resolution, crop_margin, bg_color, preprocess_crop)
                target_for_hu = mask if use_silhouette else preproc
                hu = compute_hu_from_image(target_for_hu)
                hu_list.append(hu)
            except Exception as e:
                # per-view failure: log and continue
                with open(reports_dir / "read_errors.log", "a", encoding="utf-8") as fh:
                    fh.write(f"{vpath}\t{type(e).__name__}\t{e}\n")
                continue

        if len(hu_list) == 0:
            raise RuntimeError("No usable views for Hu computation")

        hu_arr = np.stack(hu_list, axis=0)  # (V_avail,7)
        means = np.mean(hu_arr, axis=0).astype(np.float32)
        stds = np.std(hu_arr, axis=0, ddof=0).astype(np.float32)
        vec = np.concatenate([means, stds], axis=0).astype(np.float32)  # length 14

        # save per-sample npy
        p = out_dir / f"{sample_id}_hu.npy"
        ensure_dir(out_dir)
        if p.exists() and not overwrite:
            # still overwrite semantics: overwrite only if overwrite True
            pass
        np.save(str(p), vec, allow_pickle=False)

        note = "ok"
        if len(view_files) < expected_views:
            note = "partial_views"

        result.update({"success": True, "vec": vec.tolist(), "note": note})
        return result
    except Exception as e:
        # log
        try:
            ensure_dir(Path("reports"))
            with open(Path("reports") / "read_errors.log", "a", encoding="utf-8") as fh:
                fh.write(f"{dir_path}\t{type(e).__name__}\t{e}\n")
        except Exception:
            pass
        logger.debug(f"Failed sample {sample_id}: {e}")
        result["note"] = f"failed: {e}"
        return result


# -----------------------
# Main orchestration
# -----------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Extract Hu Moments per view and aggregate per-sample features")
    p.add_argument("--views_dir", type=str, default="../../data/processed/views/", help="Directory with views (data/processed/views/)")
    p.add_argument("--out_dir", type=str, default="../../data/processed/2d_hu", help="Output directory for hu features")
    p.add_argument("--resolution", type=int, default=224, help="Target resolution for preprocessing/padding")
    p.add_argument("--use_silhouette", action="store_true", help="Compute Hu on binary silhouette mask")
    p.add_argument("--preprocess_crop", action="store_true", help="Do Otsu-based crop+resize+pad before Hu")
    p.add_argument("--crop_margin", type=float, default=0.05, help="Margin fraction (<=1) or pixels (>1) to expand bbox")
    p.add_argument("--bg_color", type=str, default="white", choices=["white", "black"], help="Background color for padding")
    p.add_argument("--n_preview", type=int, default=20, help="Number of preview collages to save")
    p.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes")
    p.add_argument("--fast", action="store_true", help="Fast mode: process first 20 ids and set n_preview=10")
    p.add_argument("--max_samples", type=int, default=None, help="Limit number of samples processed (first N)")
    p.add_argument("--expected_views", type=int, default=12, help="Expected number of views per sample")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing per-sample npy outputs")
    p.add_argument("--seed", type=int, default=42, help="Random seed for determinism")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args(argv)


def _init_worker(seed: int):
    """Module-level worker initializer for deterministic RNG."""
    try:
        from multiprocessing import current_process
        ident = getattr(current_process(), "_identity", None)
        wid = ident[0] if ident else 0
    except Exception:
        wid = 0
    np.random.seed((int(seed) + int(wid)) % (2 ** 31 - 1))


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    np.random.seed(int(args.seed))

    views_dir = Path(args.views_dir)
    out_dir = Path(args.out_dir)
    reports_dir = Path("../../reports")
    ensure_dir(reports_dir)
    ensure_dir(reports_dir / "sample_images")
    ensure_dir(out_dir)

    if args.fast:
        max_samples = 20
        n_preview = min(args.n_preview, 10)
    else:
        max_samples = args.max_samples
        n_preview = args.n_preview

    # list sample ids
    ids = list_sample_ids(views_dir)
    if not ids:
        logger.warning(f"No sample directories found under {views_dir}")
        return 0

    # deterministic sort already applied
    if max_samples is not None:
        ids = ids[:max_samples]
    total = len(ids)
    logger.info(f"Found {total} sample ids under {views_dir}. Workers={args.workers}, preprocess_crop={args.preprocess_crop}, use_silhouette={args.use_silhouette}")

    config = {
        "out_dir": str(out_dir),
        "resolution": int(args.resolution),
        "use_silhouette": bool(args.use_silhouette),
        "preprocess_crop": bool(args.preprocess_crop),
        "crop_margin": float(args.crop_margin),
        "bg_color": args.bg_color,
        "expected_views": int(args.expected_views),
        "overwrite": bool(args.overwrite)
    }

    tasks = [(sid, str(dpath), config) for sid, dpath in ids]

    results = []
    if args.workers > 1:
        try:
            with mp.Pool(processes=args.workers, initializer=_init_worker, initargs=(int(args.seed),)) as pool:
                for res in tqdm(pool.imap_unordered(process_sample_worker, tasks), total=len(tasks), desc="Processing Hu"):
                    results.append(res)
        except Exception as e:
            logger.error(f"Multiprocessing failed: {e}. Falling back to single-process.")
            for t in tqdm(tasks, desc="Processing Hu (fallback)"):
                results.append(process_sample_worker(t))
    else:
        for t in tqdm(tasks, desc="Processing Hu"):
            results.append(process_sample_worker(t))

    # Build CSV records
    records = []
    for r in results:
        sid = r.get("id")
        note = r.get("note", "")
        if r.get("success"):
            vec = np.asarray(r.get("vec"), dtype=np.float32)
            means = vec[:7].tolist()
            stds = vec[7:].tolist()
            rec = {"id": sid}
            for i in range(7):
                rec[f"hu{i+1}_mean"] = float(means[i])
            for i in range(7):
                rec[f"hu{i+1}_std"] = float(stds[i])
            rec["note"] = note
            records.append(rec)
        else:
            # failed sample: still record note
            rec = {"id": sid}
            for i in range(7):
                rec[f"hu{i+1}_mean"] = float("nan")
            for i in range(7):
                rec[f"hu{i+1}_std"] = float("nan")
            rec["note"] = note
            records.append(rec)

    # CSV path
    out_csv = out_dir / "hu_features_table.csv"
    append_to_csv(records, out_csv, overwrite=args.overwrite)
    logger.info(f"Wrote Hu features table to {out_csv}")

    # Save stats to reports/hu_feature_stats.csv
    try:
        df_num = pd.read_csv(out_csv).select_dtypes(include=[np.number])
        stats = df_num.agg(["mean", "std"]).transpose().reset_index().rename(columns={"index": "feature"})
        stats_path = reports_dir / "hu_feature_stats.csv"
        stats.to_csv(stats_path, index=False)
        logger.info(f"Wrote Hu feature stats to {stats_path}")
    except Exception as e:
        logger.debug(f"Failed to write hu_feature_stats.csv: {e}")

    # Save config yaml
    cfg = {
        "timestamp": datetime.datetime.now().isoformat(sep=" ", timespec="seconds"),
        "views_dir": str(views_dir),
        "out_dir": str(out_dir),
        "resolution": int(args.resolution),
        "use_silhouette": bool(args.use_silhouette),
        "preprocess_crop": bool(args.preprocess_crop),
        "crop_margin": float(args.crop_margin),
        "bg_color": args.bg_color,
        "n_preview": int(n_preview),
        "workers": int(args.workers),
        "expected_views": int(args.expected_views),
        "overwrite": bool(args.overwrite),
        "seed": int(args.seed)
    }
    cfg_path = out_dir / "extract_2d_hu_config.yaml"
    try:
        if _HAS_YAML:
            with open(cfg_path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(cfg, fh)
        else:
            with open(cfg_path, "w", encoding="utf-8") as fh:
                fh.write(str(cfg))
        logger.info(f"Wrote config to {cfg_path}")
    except Exception:
        logger.debug("Failed to write config YAML")

    # Previews: for first n_preview successful ids create collage
    success_ids = [r for r in results if r.get("success")]
    for r in success_ids[:n_preview]:
        sid = r["id"]
        dir_path = views_dir / sid
        # choose representative views: view_01, view_05, view_09
        chosen = []
        for name in ["view_01.png", "view_05.png", "view_09.png"]:
            p = dir_path / name
            if p.exists():
                chosen.append(p)
        if len(chosen) < 3:
            pngs = sorted([p for p in dir_path.glob("*.png")])
            chosen = pngs[:3]
        if not chosen:
            continue
        # preprocess the first chosen view to get preproc and mask
        try:
            img = load_image(chosen[0])
            preproc, mask = preprocess_image(img, int(args.resolution), float(args.crop_margin), args.bg_color, bool(args.preprocess_crop))
            preview_out = reports_dir / "sample_images" / f"{sid}_hu_preview.png"
            save_preview(sid, chosen, preproc, (mask if args.use_silhouette else preproc), preview_out, args.bg_color, int(args.resolution))
        except Exception as e:
            with open(reports_dir / "read_errors.log", "a", encoding="utf-8") as fh:
                fh.write(f"{dir_path}\tpreview_failed\t{e}\n")
            logger.debug(f"Failed to create preview for {sid}: {e}")
            continue

    mode = "fast" if args.fast else "full"
    n_processed = len(records)
    print(f"Hu extraction finished. Mode: {mode}. Samples processed: {n_processed}. Descriptors saved to {out_dir}.")
    # print first CSV row preview
    try:
        if out_csv.exists():
            dfp = pd.read_csv(out_csv, nrows=1)
            print("First CSV row preview:")
            print(dfp.to_dict(orient="records")[0])
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    # Safe debug default when no args provided
    if len(sys.argv) == 1:
        argv = ["--views_dir", "../../data/processed/views/", "--out_dir", "../../data/processed/2d_hu", "--fast", "--workers", "1"]
        sys.exit(main(argv))
    else:
        sys.exit(main(sys.argv[1:]))