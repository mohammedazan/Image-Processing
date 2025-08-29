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

# optional scipy.ndimage
try:
    from scipy import ndimage as ndi
    _HAS_NDIMAGE = True
except Exception:
    ndi = None
    _HAS_NDIMAGE = False

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
        # np.fromfile allows loading paths with non-ascii on Windows
        im = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if im is None:
            pil = Image.open(str(path)).convert("L")
            return np.asarray(pil, dtype=np.uint8)
        return im.astype(np.uint8)
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
    level = 0
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
    Auto-invert mask if the foreground (white) covers >50% of image (likely background).
    Applies light Gaussian blur and morphological cleanup to reduce speckle.
    Ensures mask values are 0 or 255 uint8.
    """
    # apply small blur first to connect sparse dots (prefer cv2 GaussianBlur)
    if _HAS_CV2:
        try:
            blur = cv2.GaussianBlur(img, (5, 5), 0)
        except Exception:
            blur = img
        try:
            thresh_val, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = int(thresh_val)
        except Exception:
            t = _otsu_threshold_numpy(blur)
            thresh = int(t)
            mask = (blur > t).astype(np.uint8) * 255
    else:
        # try PIL blur fallback
        try:
            from PIL import ImageFilter
            pil = Image.fromarray(img)
            pil = pil.filter(ImageFilter.GaussianBlur(radius=1))
            blur = np.asarray(pil)
        except Exception:
            blur = img
        t = _otsu_threshold_numpy(blur)
        thresh = int(t)
        mask = (blur > t).astype(np.uint8) * 255

    # Ensure mask is binary 0/255 (uint8)
    mask = (mask > 127).astype(np.uint8) * 255

    # Heuristic invert if foreground > 50% (likely background was selected)
    fg_ratio = float(np.count_nonzero(mask) / mask.size)
    if fg_ratio > 0.5:
        mask = (255 - mask).astype(np.uint8)

    # Morphological cleanup: close then open to fill holes and remove small speckle
    if _HAS_CV2:
        try:
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern, iterations=1)
        except Exception:
            pass
    else:
        if _HAS_NDIMAGE:
            try:
                binm = (mask > 127)
                binm = ndi.binary_closing(binm, structure=np.ones((3, 3)))
                binm = ndi.binary_opening(binm, structure=np.ones((3, 3)))
                mask = (binm.astype(np.uint8) * 255)
            except Exception:
                pass

    return int(thresh), mask.astype(np.uint8)


def find_largest_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Find bounding box of largest connected component in binary mask (uint8 0/255).
    Returns x,y,w,h. If none found, return whole image bbox.
    Robust handling of cv2.findContours return signature.
    """
    h, w = mask.shape[:2]
    if _HAS_CV2:
        # cv2.findContours return signature differs between versions; handle both.
        contours_info = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours are either at index 0 or 1 depending on OpenCV version
        if len(contours_info) == 2:
            contours = contours_info[0]
        elif len(contours_info) >= 3:
            contours = contours_info[1]
        else:
            contours = []
    else:
        coords = np.argwhere(mask > 0)
        if coords.size == 0:
            return 0, 0, w, h
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        return int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)

    if not contours:
        return 0, 0, w, h
    # choose largest by area (cv2.contourArea)
    max_cnt = max(contours, key=lambda c: cv2.contourArea(c) if _HAS_CV2 else 0)
    x, y, ww, hh = cv2.boundingRect(max_cnt)
    return int(x), int(y), int(ww), int(hh)


def pad_and_resize(img: np.ndarray, target: int, bg_color: str) -> np.ndarray:
    """
    Resize preserving aspect ratio and pad to (target,target) with bg_color.
    img is numpy uint8 grayscale. Return uint8 grayscale array shape (target,target).
    """
    pil = Image.fromarray(img)  # 'L'
    # maintain aspect ratio and fit inside target x target
    pil_thumb = ImageOps.contain(pil, (target, target))
    # create background in 'L' mode (grayscale) to avoid accidental color conversions
    bg = 255 if bg_color == "white" else 0
    new_im = Image.new("L", (target, target), color=bg)
    # paste centered
    paste = pil_thumb.convert("L")
    x = (target - paste.width) // 2
    y = (target - paste.height) // 2
    new_im.paste(paste, (x, y))
    return np.asarray(new_im, dtype=np.uint8)


# -----------------------
# Mask cleaning helper
# -----------------------
def clean_mask_np(mask_np: np.ndarray, resolution: int, min_size: Optional[int] = None) -> np.ndarray:
    """
    Clean binary mask (uint8 0/255) and return cleaned mask (uint8 0/255).
    Steps:
      - ensure binary
      - morphological opening (cv2 or scipy.ndimage), fallback to simple numpy op
      - remove small components (keep >= min_size), if none keep largest
    """
    if min_size is None:
        min_size = max(8, int(0.001 * resolution * resolution))  # default: 0.1% of image area

    # ensure binary 0/1
    bin_mask = (mask_np > 127).astype(np.uint8)

    # morphological opening (3x3)
    if _HAS_CV2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx((bin_mask * 255).astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)
        bin_mask = (opened > 127).astype(np.uint8)
    elif _HAS_NDIMAGE:
        bin_mask = ndi.binary_opening(bin_mask, structure=np.ones((3, 3))).astype(np.uint8)
    else:
        # simple erosion then dilation (numpy)
        pad = np.pad(bin_mask, 1, mode="constant", constant_values=0)
        eroded = np.zeros_like(bin_mask)
        for y in range(bin_mask.shape[0]):
            for x in range(bin_mask.shape[1]):
                if pad[y:y+3, x:x+3].sum() == 9:
                    eroded[y, x] = 1
        pad2 = np.pad(eroded, 1, mode="constant", constant_values=0)
        dilated = np.zeros_like(eroded)
        for y in range(eroded.shape[0]):
            for x in range(eroded.shape[1]):
                if pad2[y:y+3, x:x+3].sum() >= 1:
                    dilated[y, x] = 1
        bin_mask = dilated

    # Remove small components; prefer keeping components >= min_size; if none, keep largest
    if _HAS_CV2:
        num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(bin_mask.astype(np.uint8), connectivity=8)
        if num_labels <= 1:
            cleaned = bin_mask
        else:
            areas = stats[1:, cv2.CC_STAT_AREA]
            keep = np.zeros_like(bin_mask)
            for i, area in enumerate(areas, start=1):
                if int(area) >= min_size:
                    keep[labels_im == i] = 1
            if keep.sum() == 0:
                max_idx = 1 + int(np.argmax(areas))
                keep = (labels_im == max_idx).astype(np.uint8)
            cleaned = keep
    elif _HAS_NDIMAGE:
        labeled, ncomp = ndi.label(bin_mask)
        if ncomp == 0:
            cleaned = bin_mask
        else:
            sizes = np.array(ndi.sum(bin_mask, labeled, range(1, ncomp + 1)))
            keep_flags = sizes >= min_size
            if not keep_flags.any():
                keep_flags[np.argmax(sizes)] = True
            keep_mask = np.zeros_like(bin_mask)
            for i, flag in enumerate(keep_flags, start=1):
                if flag:
                    keep_mask[labeled == i] = 1
            cleaned = keep_mask
    else:
        # naive flood fill labeling (numpy)
        h, w = bin_mask.shape
        labels = np.zeros((h, w), dtype=np.int32)
        cur = 0
        areas = []
        for y in range(h):
            for x in range(w):
                if bin_mask[y, x] and labels[y, x] == 0:
                    cur += 1
                    stack = [(y, x)]
                    labels[y, x] = cur
                    area = 0
                    while stack:
                        yy, xx = stack.pop()
                        area += 1
                        for ny in (yy-1, yy, yy+1):
                            for nx in (xx-1, xx, xx+1):
                                if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] == 0 and bin_mask[ny, nx]:
                                    labels[ny, nx] = cur
                                    stack.append((ny, nx))
                    areas.append(area)
        if cur == 0:
            cleaned = bin_mask
        else:
            areas = np.array(areas)
            keep_mask = np.zeros_like(bin_mask)
            for lbl_idx, area in enumerate(areas, start=1):
                if int(area) >= min_size:
                    keep_mask[labels == lbl_idx] = 1
            if keep_mask.sum() == 0:
                keep_mask = (labels == (1 + int(np.argmax(areas)))).astype(np.uint8)
            cleaned = keep_mask

    return (cleaned.astype(np.uint8) * 255)


# -----------------------
# Preprocessing
# -----------------------
def preprocess_image(img: np.ndarray,
                     resolution: int,
                     crop_margin: float,
                     bg_color: str,
                     preprocess_crop: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess a grayscale uint8 image:
    - If preprocess_crop: compute Otsu silhouette, bbox of largest component, expand by crop_margin,
      crop, resize+pad to resolution.
    - Returns (preprocessed_img, cleaned_silhouette_mask) as uint8 arrays.
    """
    assert img.ndim == 2
    h, w = img.shape[:2]
    # compute Otsu mask
    try:
        _, mask = compute_otsu(img)
    except Exception:
        t = int(np.median(img))
        mask = (img > t).astype(np.uint8) * 255

    if not preprocess_crop:
        # resize/pad original image
        preproc = pad_and_resize(img, resolution, bg_color)  # uint8 L

        # Resize mask preserving aspect ratio using NEAREST (no interpolation)
        mask_pil = Image.fromarray(mask).convert("L")
        ow, oh = mask_pil.width, mask_pil.height
        scale = min(resolution / float(ow), resolution / float(oh))
        new_w = max(1, int(round(ow * scale)))
        new_h = max(1, int(round(oh * scale)))
        mask_nn = mask_pil.resize((new_w, new_h), resample=Image.NEAREST)

        # re-binarize and ensure 0/255
        mask_np = np.asarray(mask_nn, dtype=np.uint8)
        mask_np = (mask_np > 128).astype(np.uint8) * 255

        # clean mask
        mask_np = clean_mask_np(mask_np, resolution)

        # paste into background centered (L mode)
        mask_bg = Image.new("L", (resolution, resolution), color=0)
        x = (resolution - new_w) // 2
        y = (resolution - new_h) // 2
        mask_bg.paste(Image.fromarray(mask_np).convert("L"), (x, y))
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

    # Resize mask preserving aspect ratio using NEAREST
    mask_pil = Image.fromarray(cropped_mask).convert("L")
    ow, oh = mask_pil.width, mask_pil.height
    scale = min(resolution / float(ow), resolution / float(oh))
    new_w = max(1, int(round(ow * scale)))
    new_h = max(1, int(round(oh * scale)))
    mask_nn = mask_pil.resize((new_w, new_h), resample=Image.NEAREST)

    # re-binarize and ensure 0/255
    mask_np = np.asarray(mask_nn, dtype=np.uint8)
    mask_np = (mask_np > 128).astype(np.uint8) * 255

    # clean mask
    mask_np = clean_mask_np(mask_np, resolution)

    # paste to center (L mode)
    mask_bg = Image.new("L", (resolution, resolution), color=0)
    xpad = (resolution - new_w) // 2
    ypad = (resolution - new_h) // 2
    mask_bg.paste(Image.fromarray(mask_np).convert("L"), (xpad, ypad))
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
        try:
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
            hu = np.zeros((7,), dtype=np.float64)
        except Exception:
            hu = np.zeros((7,), dtype=np.float64)
    # apply log transform
    eps = 1e-300
    hu_log = []
    for m in hu:
        val = -np.sign(m) * np.log10(abs(float(m)) + eps)
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
      - left: grid of chosen raw views (up to 3). Each thumbnail is exactly (resolution,resolution).
      - center: preprocessed image (exactly resolution x resolution)
      - right: cleaned mask or grayscale representation (exactly resolution x resolution)
    This enforces fixed panel widths to avoid seams/artifacts due to variable widths/padding.
    """
    # prepare fixed-size thumbnails for left (each exactly resolution x resolution, RGB)
    left_imgs = []
    for p in chosen_view_paths[:3]:
        try:
            raw = Image.open(p).convert("RGB")
            # contain then paste on fixed thumbnail canvas to ensure exact size
            thumb = ImageOps.contain(raw, (resolution, resolution))
            bg_rgb = (255, 255, 255) if bg_color == "white" else (0, 0, 0)
            thumb_canvas = Image.new("RGB", (resolution, resolution), color=bg_rgb)
            x = (resolution - thumb.width) // 2
            y = (resolution - thumb.height) // 2
            thumb_canvas.paste(thumb.convert("RGB"), (x, y))
            left_imgs.append(thumb_canvas)
        except Exception:
            bg_rgb = (255, 255, 255) if bg_color == "white" else (0, 0, 0)
            left_imgs.append(Image.new("RGB", (resolution, resolution), color=bg_rgb))

    # if no left imgs, create one blank thumbnail
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

    # center: ensure preproc_img is exactly (resolution,resolution) L mode then convert to RGB
    try:
        center_pil = Image.fromarray(preproc_img).convert("L")
        if center_pil.size != (resolution, resolution):
            center_pil = center_pil.resize((resolution, resolution), resample=Image.BICUBIC)
        center = center_pil.convert("RGB")
    except Exception:
        center = Image.new("RGB", (resolution, resolution), color=(255, 255, 255) if bg_color == "white" else (0, 0, 0))

    # right: mask or grayscale (ensure binary mask is preserved using NEAREST)
    try:
        # mask should be uint8 L (0/255)
        right_pil = Image.fromarray(mask).convert("L")
        if right_pil.size != (resolution, resolution):
            right_pil = right_pil.resize((resolution, resolution), resample=Image.NEAREST)
        # ensure strict binary after any resizing
        right_pil = right_pil.point(lambda p: 255 if p > 128 else 0).convert("L")
        right = right_pil.convert("RGB")
    except Exception:
        right = Image.new("RGB", (resolution, resolution), color=(255, 255, 255) if bg_color == "white" else (0, 0, 0))

    # normalize heights (should be equal because thumbnails fixed, but keep safe)
    h = max(left.height, center.height, right.height)

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
    reports_dir = Path(config.get("reports_dir", "reports"))  # use config-provided reports_dir
    result = {"success": False, "id": sample_id, "vec": None, "note": ""}

    try:
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
                try:
                    ensure_dir(reports_dir)
                    with open(reports_dir / "read_errors.log", "a", encoding="utf-8") as fh:
                        fh.write(f"{vpath}\t{type(e).__name__}\t{e}\n")
                except Exception:
                    pass
                continue

        if len(hu_list) == 0:
            raise RuntimeError("No usable views for Hu computation")

        hu_arr = np.stack(hu_list, axis=0)  # (V_avail,7)
        means = np.mean(hu_arr, axis=0).astype(np.float32)
        stds = np.std(hu_arr, axis=0, ddof=0).astype(np.float32)
        vec = np.concatenate([means, stds], axis=0).astype(np.float32)  # length 14

        p = out_dir / f"{sample_id}_hu.npy"
        ensure_dir(out_dir)
        if p.exists() and not overwrite:
            pass
        np.save(str(p), vec, allow_pickle=False)

        note = "ok"
        if len(view_files) < expected_views:
            note = "partial_views"

        result.update({"success": True, "vec": vec.tolist(), "note": note})
        return result
    except Exception as e:
        try:
            ensure_dir(Path(config.get("reports_dir", "reports")))
            with open(Path(config.get("reports_dir", "reports")) / "read_errors.log", "a", encoding="utf-8") as fh:
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

    ids = list_sample_ids(views_dir)
    if not ids:
        logger.warning(f"No sample directories found under {views_dir}")
        return 0

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
        "overwrite": bool(args.overwrite),
        "reports_dir": str(reports_dir)   # pass reports_dir explicitly
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
            rec = {"id": sid}
            for i in range(7):
                rec[f"hu{i+1}_mean"] = float("nan")
            for i in range(7):
                rec[f"hu{i+1}_std"] = float("nan")
            rec["note"] = note
            records.append(rec)

    out_csv = out_dir / "hu_features_table.csv"
    append_to_csv(records, out_csv, overwrite=args.overwrite)
    logger.info(f"Wrote Hu features table to {out_csv}")

    try:
        df_num = pd.read_csv(out_csv).select_dtypes(include=[np.number])
        stats = df_num.agg(["mean", "std"]).transpose().reset_index().rename(columns={"index": "feature"})
        stats_path = reports_dir / "hu_feature_stats.csv"
        stats.to_csv(stats_path, index=False)
        logger.info(f"Wrote Hu feature stats to {stats_path}")
    except Exception as e:
        logger.debug(f"Failed to write hu_feature_stats.csv: {e}")

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

    # Previews: for first n_preview successful ids create collage and print before/after white pixel counts
    success_ids = [r for r in results if r.get("success")]
    preview_debug_printed = 0
    for r in success_ids[:n_preview]:
        sid = r["id"]
        dir_path = views_dir / sid
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
        try:
            img = load_image(chosen[0])
            # raw mask before cleaning (for stats)
            try:
                _, raw_mask = compute_otsu(img)
            except Exception:
                t = int(np.median(img))
                raw_mask = (img > t).astype(np.uint8) * 255

            preproc, mask = preprocess_image(img, int(args.resolution), float(args.crop_margin), args.bg_color, bool(args.preprocess_crop))
            preview_out = reports_dir / "sample_images" / f"{sid}_hu_preview.png"

            # pass mask if use_silhouette else pass preproc (kept behavior) but both are forced to have correct shapes inside save_preview
            save_preview(sid, chosen, preproc, (mask if args.use_silhouette else preproc), preview_out, args.bg_color, int(args.resolution))

            if preview_debug_printed < 5:
                before = int(np.count_nonzero(raw_mask))
                after = int(np.count_nonzero(mask))
                print(f"[mask-clean] {sid}: white-pixels before={before}, after={after}")
                preview_debug_printed += 1

        except Exception as e:
            try:
                with open(reports_dir / "read_errors.log", "a", encoding="utf-8") as fh:
                    fh.write(f"{dir_path}\tpreview_failed\t{e}\n")
            except Exception:
                pass
            logger.debug(f"Failed to create preview for {sid}: {e}")
            continue

    mode = "fast" if args.fast else "full"
    n_processed = len(records)
    print(f"Hu extraction finished. Mode: {mode}. Samples processed: {n_processed}. Descriptors saved to {out_dir}.")
    try:
        if out_csv.exists():
            dfp = pd.read_csv(out_csv, nrows=1)
            print("First CSV row preview:")
            print(dfp.to_dict(orient="records")[0])
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        argv = ["--views_dir", "../../data/processed/views/", "--out_dir", "../../data/processed/2d_hu", "--fast", "--workers", "1"]
        sys.exit(main(argv))
    else:
        sys.exit(main(sys.argv[1:]))
