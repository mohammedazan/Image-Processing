#!/usr/bin/env python3
"""
train_pointnet.py

Train a PointNet model on 3D point cloud datasets.

Supports --labels_csv: CSV with columns ['id','label'] to explicitly map dataset files to labels.
Robust stratified selection for --fast and fallback when classes are too few for stratified splits.
"""
from __future__ import annotations
import argparse
import logging
import os
import random
import shutil
import csv
import json
import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    # for nullcontext on older python
    from contextlib import nullcontext
except Exception:
    from contextlib import contextmanager
    @contextmanager
    def nullcontext():
        yield

# -------------------------
# Logger setup
# -------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# -------------------------
# Helpers for id normalization (same spirit as C1)
# -------------------------
def _normalize_id(s: object) -> str:
    s = str(s)
    s = re.sub(r'\.(pts|xyz|ply|pcd|npy|txt|csv)$', '', s, flags=re.IGNORECASE)
    s = s.replace('\\', '/')
    s = s.replace('/', '_')
    s = re.sub(r'[\s\-\:]+', '_', s)
    s = re.sub(r'_+', '_', s)
    return s.strip('_')


def coerce_id_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    if 'id' not in df.columns:
        for cand in ['Id', 'ID']:
            if cand in df.columns:
                df = df.rename(columns={cand: 'id'})
                break
    if 'label' not in df.columns:
        for cand in ['Label', 'LABEL']:
            if cand in df.columns:
                df = df.rename(columns={cand: 'label'})
                break
    return df


# -------------------------
# Small utility: robust stratified selection (used for FAST)
# -------------------------
def stratified_select(files: List[Path], labels: List[str], n_select: int, seed: int) -> Tuple[List[Path], List[str]]:
    """Try StratifiedShuffleSplit to pick n_select samples. If impossible (rare), do manual stratified-ish selection:
       - ensure at least one sample per class when possible
       - then fill remaining slots randomly from remaining indices
    Returns selected_files, selected_labels (preserving parallel order).
    """
    n_total = len(files)
    if n_select >= n_total:
        return list(files), list(labels)

    rng = np.random.RandomState(seed)
    try:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_select, random_state=seed)
        sel_idx, _ = next(sss.split(files, labels))
        sel_idx = list(sel_idx)
        selected_files = [files[i] for i in sel_idx]
        selected_labels = [labels[i] for i in sel_idx]
        return selected_files, selected_labels
    except Exception:
        logger.warning("StratifiedShuffleSplit failed for fast selection (likely very small classes). Falling back to manual stratified selection.")
        # Manual: one per class if possible
        by_class = {}
        for i, lab in enumerate(labels):
            by_class.setdefault(lab, []).append(i)
        selected = []
        # ensure at least one per class (but not exceeding budget)
        for lab, idxs in by_class.items():
            if len(selected) >= n_select:
                break
            chosen = rng.choice(idxs, 1, replace=False)
            selected.append(int(chosen[0]))
        # fill remaining randomly
        remaining = [i for i in range(n_total) if i not in selected]
        rng.shuffle(remaining)
        while len(selected) < n_select and remaining:
            selected.append(int(remaining.pop()))
        selected = selected[:n_select]
        selected_files = [files[i] for i in selected]
        selected_labels = [labels[i] for i in selected]
        return selected_files, selected_labels


# -------------------------
# Dataset
# -------------------------
class PointCloudDataset(Dataset):
    def __init__(self, root_dir: Optional[str] = None, files: Optional[List[Path]] = None,
                 augment: bool = False, label2idx: Optional[dict] = None,
                 npoints: int = 1024, seed: int = 42):
        if files is not None:
            self.files = [Path(f) for f in sorted(files)]
        elif root_dir is not None:
            self.files = sorted(Path(root_dir).glob("*.npy"))
        else:
            raise ValueError("Either root_dir or files must be provided")

        self.augment = augment
        self.npoints = int(npoints)
        self.seed = int(seed)

        if label2idx is None:
            labels = set(Path(f).stem.split("_")[0] for f in self.files)
            self.label2idx = {l: i for i, l in enumerate(sorted(labels))}
        else:
            self.label2idx = label2idx

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        pts = np.load(fpath).astype(np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 3)
        if pts.shape[-1] != 3:
            if pts.shape[0] == 3 and pts.shape[1] != 3:
                pts = pts.T
            else:
                raise ValueError(f"Point cloud {fpath} has unsupported shape {pts.shape}")

        rng = np.random.RandomState(self.seed + int(idx))
        m = pts.shape[0]
        if m >= self.npoints:
            choice = rng.choice(m, self.npoints, replace=False)
            pts = pts[choice, :]
        else:
            needed = self.npoints - m
            if m > 0:
                pad_idx = rng.choice(m, needed, replace=True)
                pads = pts[pad_idx, :]
                pts = np.vstack([pts, pads])
            else:
                pts = np.zeros((self.npoints, 3), dtype=np.float32)

        label_str = Path(fpath).stem.split("_")[0]
        if label_str not in self.label2idx:
            raise KeyError(f"Label '{label_str}' from file {fpath} not found in label2idx mapping.")
        label = self.label2idx[label_str]

        if self.augment:
            pts = self.apply_augmentations(pts, rng)

        return torch.from_numpy(pts), torch.tensor(label, dtype=torch.long)

    def apply_augmentations(self, pts: np.ndarray, rng: np.random.RandomState):
        theta = rng.uniform(0, 2 * np.pi)
        cosval = np.cos(theta)
        sinval = np.sin(theta)
        rot = np.array([[cosval, -sinval, 0],
                        [sinval,  cosval, 0],
                        [0,       0,      1]], dtype=np.float32)
        pts = pts @ rot.T
        scale = rng.uniform(0.8, 1.2)
        pts = pts * float(scale)
        noise = np.clip(0.01 * rng.randn(*pts.shape), -0.05, 0.05).astype(np.float32)
        pts = pts + noise
        if rng.rand() < 0.3:
            num_drop = max(1, len(pts) // 10)
            drop_idx = rng.choice(len(pts), num_drop, replace=False)
            pts[drop_idx] = pts[0]
        shift = rng.uniform(-0.1, 0.1, (1, 3)).astype(np.float32)
        pts = pts + shift
        return pts


# -------------------------
# Model: PointNet
# -------------------------
class PointNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x


# -------------------------
# Evaluation
# -------------------------
def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for pts, labels in loader:
            pts = pts.to(device, dtype=torch.float32)
            labels = labels.to(device)
            outputs = model(pts)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * pts.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, avg_loss


# -------------------------
# Prepare files & labels
# -------------------------
def prepare_file_label_lists(data_dir: str, labels_csv: Optional[str] = None) -> Tuple[List[Path], List[str], Dict[str, Path]]:
    root = Path(data_dir)
    files = sorted(root.glob("*.npy"))
    if not files:
        raise SystemExit(f"No .npy files found in {data_dir}")

    file_map: Dict[str, Path] = {}
    for p in files:
        stem = _normalize_id(p.stem)
        if stem not in file_map:
            file_map[stem] = p

    if labels_csv and Path(labels_csv).exists():
        df = pd.read_csv(labels_csv)
        df = coerce_id_label_columns(df)
        if not {'id', 'label'}.issubset(df.columns):
            raise ValueError(f"Labels CSV {labels_csv} must contain columns 'id' and 'label'")
        df['id_norm'] = df['id'].astype(str).map(_normalize_id)
        matched_files = []
        matched_labels = []
        missing_ids = []
        for _, row in df.iterrows():
            idn = row['id_norm']
            lbl = str(row['label'])
            if idn in file_map:
                matched_files.append(file_map[idn])
                matched_labels.append(lbl)
            else:
                missing_ids.append(row['id'])
        if len(missing_ids) > 0:
            logger.warning(f"{len(missing_ids)} ids from labels CSV not found among .npy files. Examples: {missing_ids[:5]}")
        unlabeled_files = [p for k, p in file_map.items() if k not in set(df['id_norm'].tolist())]
        if unlabeled_files:
            logger.info(f"{len(unlabeled_files)} files in {data_dir} have no matching label in {labels_csv} and will be ignored for training.")
        return matched_files, matched_labels, file_map
    else:
        inferred_labels = [Path(p).stem.split("_")[0] for p in files]
        return files, inferred_labels, file_map


# -------------------------
# Main train function
# -------------------------
def train_pointnet(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files, all_label_strs, file_map = prepare_file_label_lists(args.data_dir, args.labels_csv)
    n_samples = len(all_files)
    if n_samples == 0:
        raise SystemExit("No training samples after alignment with labels (maybe mismatch of ids).")

    labels_sorted = sorted(list(set(all_label_strs)))
    label2idx = {l: i for i, l in enumerate(labels_sorted)}
    num_classes = len(label2idx)
    logger.info(f"Prepared {n_samples} samples, {num_classes} classes.")

    # FAST mode: stratified selection with fallback
    if args.fast:
        n_fast = min(20, n_samples)
        if n_fast < n_samples:
            selected_files, selected_labels = stratified_select(all_files, all_label_strs, n_fast, args.seed)
            all_files = selected_files
            all_label_strs = selected_labels
            logger.info(f"FAST mode: selected {len(all_files)} stratified samples for debug.")
        else:
            logger.info("FAST mode: dataset smaller than fast-size, using full set.")

    # For train/val split: attempt stratified only if every class has >=2 samples
    cnts = Counter(all_label_strs)
    min_count = min(cnts.values()) if cnts else 0
    logger.debug(f"Class counts after FAST (if used): {dict(cnts)}")
    val_fraction = max(0.01, min(0.5, args.val_fraction))

    if min_count >= 2:
        # stratified split safe
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=args.seed)
        train_idx, val_idx = next(sss.split(all_files, [label2idx[s] for s in all_label_strs]))
        logger.info("Using stratified train/val split.")
    else:
        # fallback to random split (not stratified)
        logger.warning("Some classes have <2 samples — falling back to non-stratified random split for train/val.")
        ss = ShuffleSplit(n_splits=1, test_size=val_fraction, random_state=args.seed)
        train_idx, val_idx = next(ss.split(all_files))
    train_files = [all_files[i] for i in train_idx]
    val_files = [all_files[i] for i in val_idx]
    train_labels = [all_label_strs[i] for i in train_idx]
    val_labels = [all_label_strs[i] for i in val_idx]

    # Create dataset instances
    train_dataset = PointCloudDataset(files=train_files, augment=True, label2idx=label2idx, npoints=args.npoints, seed=args.seed)
    val_dataset = PointCloudDataset(files=val_files, augment=False, label2idx=label2idx, npoints=args.npoints, seed=args.seed)
    logger.info(f"Dataset sizes -> train: {len(train_dataset)}, val: {len(val_dataset)}")

    num_workers = min(4, (os.cpu_count() or 1))
    pin_memory = (device.type == "cuda")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    model = PointNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    if device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        autocast_ctx = torch.cuda.amp.autocast
    else:
        scaler = None
        autocast_ctx = nullcontext

    best_acc = 0.0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for pts, labels in train_loader:
            pts = pts.to(device, dtype=torch.float32)
            labels = labels.to(device)
            optimizer.zero_grad()

            with autocast_ctx():
                outputs = model(pts)
                loss = criterion(outputs, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * pts.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total if total > 0 else 0.0
        train_acc = correct / total if total > 0 else 0.0

        val_acc, val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        logger.info(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        history.append((epoch, train_loss, train_acc, val_loss, val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_full = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "label2idx": label2idx,
                "args": vars(args),
            }
            torch.save(checkpoint_full, out_dir / "checkpoint_full.pth")
            torch.save(model.state_dict(), out_dir / "checkpoint.pth")
            logger.info(f"  -> New best model saved (val_acc={best_acc:.4f}).")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            logger.info("Early stopping triggered.")
            break

    torch.save(model.state_dict(), out_dir / "last_checkpoint.pth")
    logger.info(f"Training finished. Best val_acc={best_acc:.4f}")

    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for row in history:
            writer.writerow(row)

    with open(out_dir / "label2idx.json", "w", encoding="utf-8") as fh:
        json.dump(label2idx, fh, indent=2, ensure_ascii=False)

    logger.info(f"Saved metrics -> {metrics_path}")
    logger.info(f"Saved label2idx -> {out_dir / 'label2idx.json'}")
    logger.info(f"Saved checkpoints -> {out_dir / 'checkpoint_full.pth'} and {out_dir / 'checkpoint.pth'}")


# -------------------------
# Argument parser
# -------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Train PointNet on point clouds (.npy files). Filenames must start with label: <label>_<id>.npy OR use --labels_csv")
    p.add_argument("--data_dir", type=str, required=True, help="Directory with .npy point clouds")
    p.add_argument("--labels_csv", type=str, default=None, help="Optional CSV with columns 'id,label' to align files to labels")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for experiments/checkpoints")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--fast", action="store_true", help="Quick debug: stratified small subset")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    p.add_argument("--npoints", type=int, default=1024, help="Number of points per cloud (pad/trim to this)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--val_fraction", type=float, default=0.2, help="Fraction of data to use for validation (stratified)")
    return p


# -------------------------
# Main
# -------------------------
def main(argv=None):
    args = build_argparser().parse_args(argv)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    train_pointnet(args)
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        demo_argv = [
            "--data_dir", "data/processed/sampled",
            "--out_dir", "experiments/pointnet",
            "--fast", "--verbose",
            "--epochs", "5"
        ]
        raise SystemExit(main(demo_argv))
    else:
        raise SystemExit(main())
