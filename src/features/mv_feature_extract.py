#!/usr/bin/env python3
"""
mv_feature_extract.py

Multi-view CNN feature extractor (optional backbone).

For each sample directory under --views_dir (expected structure:
  <views_dir>/<sample_id>/view_01.png ... view_12.png)
the script:
  - loads V views (any number found, sorted lexicographically),
  - preprocesses images with ImageNet normalization,
  - runs a frozen ResNet50 (pretrained) and extracts the 2048-d avgpool features,
  - aggregates across views (mean and std) producing a 4096-d vector (mean || std),
  - saves per-sample numpy: <out_dir>/<id>_mv.npy (float32),
  - appends/updates a global CSV table: <out_dir>/mv_features_table.csv

Designed to be robust (CPU/GPU, missing views, small batches).
"""

from __future__ import annotations
import argparse
import datetime
import logging
import math
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# Torch & torchvision (optional GPU)
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# YAML config optional
try:
    import yaml
    _HAS_YAML = True
except Exception:
    yaml = None
    _HAS_YAML = False

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mv_feature_extract")


# -------------------------
# Helpers
# -------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_sample_dirs(views_dir: Path) -> List[Tuple[str, Path]]:
    """
    Return list of (id, path) for immediate subdirectories of views_dir that contain image files.
    """
    ids = []
    if not views_dir.exists():
        return ids
    for d in sorted(views_dir.iterdir(), key=lambda p: str(p)):
        if d.is_dir():
            imgs = sorted([p for p in d.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
            if imgs:
                ids.append((d.name, d))
    return ids


def load_and_preprocess_image(path: Path, transform) -> torch.Tensor:
    """
    Load a single image as PIL, apply transform -> tensor on CPU.
    """
    with Image.open(str(path)) as im:
        im = im.convert("RGB")
        return transform(im)  # CxHxW tensor


def build_resnet50_feature_extractor(device: torch.device, pretrained: bool = True) -> nn.Module:
    """
    Build ResNet50 and return a feature extractor module that outputs 2048-d vectors per image.
    We take output after the global avgpool and flatten it.
    """
    model = models.resnet50(pretrained=pretrained)
    # Remove fc, keep until avgpool
    # One convenient way: replace fc with Identity and on forward take the flattened avgpool output
    model.fc = nn.Identity()
    # We'll create a wrapper to forward x through model and then flatten (model returns [B,2048,1,1]??).
    class FeatureWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x):
            # base returns features before fc if fc is Identity; with torchvision ResNet, avgpool returns (B, 2048, 1, 1)
            # but after replacing fc with Identity, forward still applies avgpool and flatten in the original implementation?
            # Safer: replicate the forward up to avgpool:
            # Use base.conv1.. base.layer4 .. base.avgpool
            x = self.base.conv1(x)
            x = self.base.bn1(x)
            x = self.base.relu(x)
            x = self.base.maxpool(x)
            x = self.base.layer1(x)
            x = self.base.layer2(x)
            x = self.base.layer3(x)
            x = self.base.layer4(x)
            x = self.base.avgpool(x)      # shape (B, 2048, 1, 1)
            x = torch.flatten(x, 1)       # shape (B, 2048)
            return x

    wrapper = FeatureWrapper(model)
    wrapper.to(device)
    wrapper.eval()
    # freeze params
    for p in wrapper.parameters():
        p.requires_grad = False
    return wrapper


def append_or_update_csv(records: List[Dict[str, Any]], csv_path: Path, overwrite: bool = False) -> None:
    """
    Append records (list of dicts) to csv_path. If overwrite True, write fresh file.
    If csv exists and not overwrite, keep last entry per id.
    """
    if not records:
        return
    df_new = pd.DataFrame(records)
    if csv_path.exists() and not overwrite:
        try:
            df_old = pd.read_csv(csv_path)
            combined = pd.concat([df_old, df_new], ignore_index=True)
            combined = combined.drop_duplicates(subset=["id"], keep="last")
            combined.to_csv(csv_path, index=False)
            return
        except Exception as e:
            logger.debug(f"Failed to append to existing CSV, will overwrite. Error: {e}")
    df_new.to_csv(csv_path, index=False)


# -------------------------
# Main processing for one sample
# -------------------------
def process_sample(sample_id: str,
                   sample_dir: Path,
                   out_dir: Path,
                   model: torch.nn.Module,
                   device: torch.device,
                   transform,
                   batch_size: int,
                   expected_views: int,
                   overwrite: bool) -> Dict[str, Any]:
    """
    Process one sample directory: load views, run model in batches, aggregate mean/std, save npy.
    Returns record dict for CSV.
    """
    rec = {"id": sample_id, "note": "", "n_views": 0}
    view_files = sorted([p for p in sample_dir.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")], key=lambda p: str(p))
    if len(view_files) == 0:
        rec["note"] = "no_views"
        return rec

    tensors = []
    # load & transform on CPU
    for p in view_files:
        try:
            t = load_and_preprocess_image(p, transform)  # CPU tensor
            tensors.append(t)
        except Exception as e:
            logger.debug(f"Failed to load {p}: {e}")
            continue

    if len(tensors) == 0:
        rec["note"] = "failed_loads"
        return rec

    # stack into batches
    dataset = torch.stack(tensors, dim=0)  # (V, C, H, W)
    n_views = dataset.shape[0]
    rec["n_views"] = int(n_views)

    feats = []
    with torch.no_grad():
        for i in range(0, n_views, batch_size):
            batch = dataset[i:i+batch_size].to(device, non_blocking=True)
            out = model(batch)  # (B, feat_dim)
            out = out.detach().cpu().numpy()
            feats.append(out)
    feats_arr = np.vstack(feats)  # (n_views, feat_dim)
    if feats_arr.ndim != 2:
        rec["note"] = "bad_feat_shape"
        return rec

    # aggregate
    mean_vec = np.mean(feats_arr, axis=0).astype(np.float32)
    std_vec = np.std(feats_arr, axis=0, ddof=0).astype(np.float32)
    feat_vec = np.concatenate([mean_vec, std_vec], axis=0)  # (2*feat_dim,)

    # save per-sample npy
    ensure_dir(out_dir)
    out_path = out_dir / f"{sample_id}_mv.npy"
    if out_path.exists() and not overwrite:
        rec["note"] = "exists"
    else:
        np.save(str(out_path), feat_vec.astype(np.float32), allow_pickle=False)
        rec["note"] = "ok"

    # if fewer views than expected, set partial_views note
    if n_views < expected_views:
        rec["note"] = rec["note"] + (";partial_views" if rec["note"] else "partial_views")

    # Also put some metadata
    rec["feat_dim"] = int(mean_vec.shape[0])
    rec["saved_path"] = str(out_path)
    return rec


# -------------------------
# Multiprocessing worker (module-level for Windows pickling)
# -------------------------
def mp_worker(task: Tuple[str, str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Worker executed in a separate process. Rebuilds model and transform locally on CPU.
    Expects task = (sample_id, sample_dir_str, cfg_dict)
    cfg keys: out_dir, pretrained, batch_size, expected_views, overwrite, no_normalize
    """
    sid, dpath_str, cfg = task
    dpath = Path(dpath_str)
    out_dir = Path(cfg["out_dir"])  # ensure Path inside worker

    # Build transform inside worker
    transform_list = [
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]
    if not bool(cfg.get("no_normalize", False)):
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225]))
    transform = transforms.Compose(transform_list)

    # Rebuild model inside worker (CPU)
    dev = torch.device("cpu")
    local_model = build_resnet50_feature_extractor(dev, pretrained=bool(cfg.get("pretrained", False)))

    return process_sample(
        sid,
        dpath,
        out_dir,
        local_model,
        dev,
        transform,
        int(cfg.get("batch_size", 8)),
        int(cfg.get("expected_views", 12)),
        bool(cfg.get("overwrite", False))
    )

# -------------------------
# CLI / main
# -------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Multi-view CNN feature extraction (ResNet50 backbone).")
    p.add_argument("--views_dir", type=str, default="../../data/processed/views/", help="Directory with rendered views (per-sample subdirs)")
    p.add_argument("--out_dir", type=str, default="../../data/processed/mv_features", help="Output directory for multi-view features")
    p.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda (e.g. cuda:0)")
    p.add_argument("--batch_size", type=int, default=8, help="Batch size for model inference (per-sample views)")
    p.add_argument("--workers", type=int, default=1, help="Parallel worker processes (multiprocessing pool). 1 = single-process")
    p.add_argument("--pretrained", action="store_true", help="Use pretrained ResNet50 weights (default False for explicitness); set if you want pretrained features")
    p.add_argument("--expected_views", type=int, default=12, help="Expected number of views per sample (for notes)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing per-sample .npy files")
    p.add_argument("--no_normalize", action="store_true", help="Disable ImageNet normalization (not recommended)")
    p.add_argument("--workers_img", type=int, default=0, help="Number of threads for image loading (unused default).")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args(argv)


def save_labels_from_ids(ids: List[str], out_csv: Path) -> None:
    """
    Create a labels CSV from sample IDs (id,label).
    Label is inferred as prefix before first underscore.
    """
    rows = []
    for sid in ids:
        label = sid.split("_")[0] if "_" in sid else sid
        rows.append({"id": sid, "label": label})

    df = pd.DataFrame(rows).drop_duplicates()
    ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved derived labels CSV: {out_csv} ({len(df)} rows)")


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    views_dir = Path(args.views_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    sample_list = list_sample_dirs(views_dir)
    if not sample_list:
        logger.warning(f"No sample directories with images found under {views_dir}")
        return 0

    # device handling
    device_str = args.device
    if device_str.lower().startswith("cuda") and torch.cuda.is_available():
        device = torch.device(device_str)
    else:
        device = torch.device("cpu")
        if device_str.lower().startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")

    # transforms (ImageNet defaults)
    transform_list = [
        transforms.Resize((224, 224)),  # ensure consistent input size
        transforms.ToTensor()
    ]
    if not args.no_normalize:
        # standard ImageNet normalization
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225]))
    transform = transforms.Compose(transform_list)

    # build model (frozen)
    model = build_resnet50_feature_extractor(device=device, pretrained=bool(args.pretrained))

    logger.info(f"Device: {device}; pretrained={args.pretrained}; transform normalization={'OFF' if args.no_normalize else 'ON'}")
    logger.info(f"Found {len(sample_list)} samples under {views_dir}")

    results = []
    # Single-process loop (workers>1 optional via multiprocessing.Pool, but keep simple/reliable)
    if args.workers <= 1:
        for sid, dpath in tqdm(sample_list, desc="Extracting mv features"):
            rec = process_sample(sid, dpath, out_dir, model, device, transform, args.batch_size, args.expected_views, args.overwrite)
            results.append(rec)
    else:
        # Multiprocessing: spawn processes that each create their own model on device=cpu (avoid GPU multi-process issues)
        # For safety we force device to cpu inside multiprocessing worker
        import multiprocessing as mp

        tasks = [
            (
                sid,
                str(dpath),
                {
                    "out_dir": str(out_dir),
                    "pretrained": bool(args.pretrained),
                    "batch_size": int(args.batch_size),
                    "expected_views": int(args.expected_views),
                    "overwrite": bool(args.overwrite),
                    "no_normalize": bool(args.no_normalize),
                },
            )
            for sid, dpath in sample_list
        ]
        with mp.Pool(processes=args.workers) as pool:
            for r in tqdm(pool.imap_unordered(mp_worker, tasks), total=len(tasks), desc="Extracting mv features (mp)"):
                results.append(r)

    # Build CSV with feature metadata (we do not inline 4096 dims into CSV to avoid huge files).
    # Instead we store per-sample npy paths and metadata; but many users prefer full flattened features in CSV.
    # We'll include minimal meta and saved_path and feat_dim.
    csv_records = []
    for r in results:
        rec_csv = {"id": r.get("id", ""), "note": r.get("note", ""), "n_views": int(r.get("n_views", 0)), "feat_dim": int(r.get("feat_dim", 0)), "saved_path": r.get("saved_path", "")}
        csv_records.append(rec_csv)

    csv_path = out_dir / "mv_features_table.csv"
    append_or_update_csv(csv_records, csv_path, overwrite=args.overwrite)
    logger.info(f"Wrote mv features table to {csv_path}")

    ids_done = [r["id"] for r in results if "id" in r]
    labels_csv = out_dir / "mv_labels.csv"
    save_labels_from_ids(ids_done, labels_csv)

    # Save run config
    cfg = {
        "timestamp": datetime.datetime.now().isoformat(sep=" ", timespec="seconds"),
        "views_dir": str(views_dir),
        "out_dir": str(out_dir),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "workers": int(args.workers),
        "pretrained": bool(args.pretrained),
        "expected_views": int(args.expected_views),
        "overwrite": bool(args.overwrite),
        "no_normalize": bool(args.no_normalize),
        "seed": int(args.seed)
    }
    cfg_path = out_dir / "mv_feature_extract_config.yaml"
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

    print(f"mv feature extraction finished. Samples processed: {len(results)}. Features saved to {out_dir}.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        argv = ["--views_dir", "../../data/processed/views/", "--out_dir", "../../data/processed/mv_features", "--batch_size", "8", "--workers", "1", "--pretrained"]
        sys.exit(main(argv))
    else:
        sys.exit(main(sys.argv[1:]))
