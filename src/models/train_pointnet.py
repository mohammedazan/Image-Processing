#!/usr/bin/env python3
"""
train_pointnet.py

Train a PointNet model on 3D point cloud datasets.

Usage (debug / FAST mode):
    python train_pointnet.py --data_dir data/sampled --out_dir experiments/pointnet --fast --verbose --epochs 5

Full run example:
    python train_pointnet.py --data_dir /content/drive/MyDrive/tree_dataset/data/mv_features \
                             --out_dir /content/drive/MyDrive/tree_dataset/experiments/pointnet \
                             --epochs 50 --batch_size 32 --lr 0.001 --patience 5

"""
import argparse
import logging
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from contextlib import nullcontext

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
# Dataset
# -------------------------
class PointCloudDataset(Dataset):
    def __init__(self, root_dir, augment=False, label2idx=None):
        self.files = sorted(Path(root_dir).glob("*.npy"))
        self.augment = augment
        if label2idx is None:
            labels_set = set(Path(f).stem.split("_")[0] for f in self.files)
            self.label2idx = {l:i for i,l in enumerate(sorted(labels_set))}
        else:
            self.label2idx = label2idx

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        pts = np.load(f).astype(np.float32)
        label_str = Path(f).stem.split("_")[0]
        label = self.label2idx[label_str]
    
        if self.augment:
            pts = self.apply_augmentations(pts)
        return torch.from_numpy(pts), label
        
        
    def apply_augmentations(self, pts):
        # Rotation around Z axis
        theta = random.uniform(0, 2 * np.pi)
        rot = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ])
        pts = pts @ rot.T

        # Scaling
        scale = random.uniform(0.8, 1.2)
        pts *= scale

        # Jitter
        pts += np.clip(0.01 * np.random.randn(*pts.shape), -0.05, 0.05)

        # Dropout points (simulate occlusion)
        if random.random() < 0.3:
            drop_idx = np.random.choice(len(pts), len(pts) // 10, replace=False)
            pts[drop_idx] = pts[0]

        # Shift
        pts += np.random.uniform(-0.1, 0.1, 3)

        return pts

# -------------------------
# Model: PointNet
# -------------------------
class PointNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = x.transpose(2, 1)  # (B, 3, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]  # Global max pooling
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.dropout(self.fc2(x))))
        return self.fc3(x)

# -------------------------
# Training utilities
# -------------------------
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for pts, labels in loader:
            pts, labels = pts.to(device), labels.to(device)
            outputs = model(pts)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * pts.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total, total_loss / total

# -------------------------
# Training
# -------------------------
def train_pointnet(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Dataset split
    tmp_files = sorted(Path(args.data_dir).glob("*.npy"))
    labels_set = set(Path(f).stem.split("_")[0] for f in tmp_files)
    label2idx = {l:i for i,l in enumerate(sorted(labels_set))}
    
    full_dataset = PointCloudDataset(args.data_dir, augment=True, label2idx=label2idx)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.augment = False

    num_workers = min(2, os.cpu_count())
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    # Model
    def get_label_from_filename(f):
        name = Path(f).stem  # 'Buche_12'
        label_str = name.split("_")[0]  # 'Buche'
        return label_str

    
    def get_label(f):
        data = np.load(f, allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile):
            return int(data['label'])
        elif isinstance(data, np.ndarray) and data.dtype == object:
            
            return int(data.item()['label'])
        else:
            raise ValueError(f"Unsupported file format: {f}")

    labels_set = set(get_label_from_filename(f) for f in full_dataset.files)
    num_classes = len(labels_set)
    label2idx = {l:i for i,l in enumerate(sorted(labels_set))}
    
    model = PointNet(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # AMP setup
    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
        autocast_ctx = torch.amp.autocast("cuda")
    else:
        scaler = None
        autocast_ctx = nullcontext()

    # Training loop
    best_acc, patience_counter = 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for pts, labels in train_loader:
            pts, labels = pts.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast_ctx:
                outputs = model(pts)
                loss = criterion(outputs, labels)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * pts.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_acc, val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        logger.info(f"Epoch {epoch}/{args.epochs}: train_loss={total_loss/total:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), Path(args.out_dir) / "best_checkpoint.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            logger.info("Early stopping triggered!")
            break

    torch.save(model.state_dict(), Path(args.out_dir) / "last_checkpoint.pth")
    logger.info(f"Training finished. Best val acc={best_acc:.4f}")

# -------------------------
# Argparser
# -------------------------
def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--fast", action="store_true", help="Load only 20 samples (debug mode)")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return p

# -------------------------
# Main
# -------------------------
def main(argv=None):
    args = build_argparser().parse_args(argv)

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.fast:
        root = Path(args.data_dir)
        files = sorted(root.glob("*.npy"))[:20]
        tmp_dir = root / "_fast_debug"
        tmp_dir.mkdir(exist_ok=True)
        for f in files:
            target = tmp_dir / f.name
            if not target.exists():
                shutil.copy(f, target)
        args.data_dir = str(tmp_dir)
        logger.info(f"FAST mode activé: {len(files)} fichiers copiés dans {tmp_dir}")

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
