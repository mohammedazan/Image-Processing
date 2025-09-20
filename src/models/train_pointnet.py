#!/usr/bin/env python3
"""
train_pointnet_dgcnn.py

Improved training script that supports PointNet (original) or DGCNN (EdgeConv) architectures.
- Use --arch {pointnet,dgcnn}
- Stronger augmentations, normalization, sampler/class-weights, OneCycleLR, EMA, TTA, mixup
"""
from __future__ import annotations
import argparse, logging, os, random, json, re
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Progress bar optional
try:
    from tqdm import trange, tqdm
except Exception:
    trange = range
    tqdm = lambda x, **kw: x

# nullcontext fallback
try:
    from contextlib import nullcontext
except Exception:
    from contextlib import contextmanager
    @contextmanager
    def nullcontext():
        yield

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("train_pointnet_dgcnn")

# -------------------------
# Helpers
# -------------------------
def _normalize_id(s: object) -> str:
    s = str(s)
    s = re.sub(r'\.(pts|xyz|ply|pcd|npy|txt|csv)$', '', s, flags=re.IGNORECASE)
    s = s.replace('\\', '/').replace('/', '_')
    s = re.sub(r'[\s\-\:]+', '_', s)
    s = re.sub(r'_+', '_', s)
    return s.strip('_')

def coerce_id_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    if 'id' not in df.columns:
        for cand in ['Id','ID']:
            if cand in df.columns:
                df = df.rename(columns={cand:'id'}); break
    if 'label' not in df.columns:
        for cand in ['Label','LABEL']:
            if cand in df.columns:
                df = df.rename(columns={cand:'label'}); break
    return df

# -------------------------
# Augmentations
# -------------------------
def rotate_axis(pts: np.ndarray, axis: str, rng: np.random.RandomState):
    if axis == 'z':
        theta = rng.uniform(0, 2*np.pi)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0],[s, c, 0],[0,0,1]], dtype=np.float32)
    else:
        theta = rng.uniform(-np.pi/6, np.pi/6)
        v = rng.normal(size=(3,))
        v = v / (np.linalg.norm(v)+1e-12)
        a = np.cos(theta/2); b, c_, d = -v*np.sin(theta/2)
        aa, bb, cc, dd = a*a, b*b, c_*c_, d*d
        bc, ad, ac, ab, bd, cd = b*c_, a*d, a*c_, a*b, b*d, c_*d
        R = np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                      [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                      [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]], dtype=np.float32)
    return pts @ R.T

def jitter(pts: np.ndarray, rng: np.random.RandomState, sigma=0.01, clip=0.03):
    noise = np.clip(sigma * rng.randn(*pts.shape), -clip, clip).astype(np.float32)
    return pts + noise

def scale_translate(pts: np.ndarray, rng: np.random.RandomState, scale_range=(0.9,1.1), shift=0.02):
    s = rng.uniform(*scale_range)
    shift_vec = rng.uniform(-shift, shift, (1,3)).astype(np.float32)
    return pts * float(s) + shift_vec

def random_point_dropout(pts: np.ndarray, rng: np.random.RandomState, max_dropout=0.2):
    if rng.rand() < 0.6:
        n = len(pts)
        drop_n = int(rng.uniform(0, max_dropout)*n)
        if drop_n > 0:
            idx = rng.choice(n, drop_n, replace=False)
            pts[idx] = pts[0]
    return pts

# -------------------------
# Dataset
# -------------------------
class PointCloudDataset(Dataset):
    def __init__(self, files: List[Path], label2idx: Dict[str,int], augment: bool=False,
                 npoints: int=2048, seed: int=42, augment_full_axes: bool=False, normalize: bool=False):
        self.files = [Path(f) for f in sorted(files)]
        self.label2idx = label2idx
        self.augment = bool(augment)
        self.npoints = int(npoints)
        self.seed = int(seed)
        self.augment_full_axes = bool(augment_full_axes)
        self.normalize = bool(normalize)

    def __len__(self):
        return len(self.files)

    def _load_pts(self, p: Path) -> np.ndarray:
        pts = np.load(p).astype(np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(-1,3)
        if pts.shape[-1] != 3:
            if pts.shape[0] == 3 and pts.shape[1] != 3:
                pts = pts.T
            else:
                raise ValueError(f"{p}: unsupported shape {pts.shape}")
        if self.normalize:
            centroid = pts.mean(axis=0, keepdims=True)
            pts = pts - centroid
            max_norm = np.max(np.linalg.norm(pts, axis=1)) if pts.shape[0] > 0 else 1.0
            pts = pts / (max_norm + 1e-12)
        return pts

    def __getitem__(self, idx):
        p = self.files[idx]
        pts = self._load_pts(p)
        rng = np.random.RandomState(self.seed + int(idx))
        m = pts.shape[0]
        if m >= self.npoints:
            choice = rng.choice(m, self.npoints, replace=False)
            pts = pts[choice,:]
        else:
            if m > 0:
                pad_n = self.npoints - m
                pad_idx = rng.choice(m, pad_n, replace=True)
                pads = pts[pad_idx,:]
                pts = np.vstack([pts, pads])
            else:
                pts = np.zeros((self.npoints,3), dtype=np.float32)

        if self.augment:
            pts = rotate_axis(pts, 'z', rng)
            if self.augment_full_axes:
                pts = rotate_axis(pts, 'any', rng)
            pts = scale_translate(pts, rng)
            pts = jitter(pts, rng, sigma=0.01, clip=0.03)
            pts = random_point_dropout(pts, rng)

        stem = p.stem
        label_str = stem.split("_")[0]
        if label_str not in self.label2idx:
            ln = _normalize_id(stem).split("_")[0]
            if ln in self.label2idx:
                label_str = ln
            else:
                if stem in self.label2idx:
                    label_str = stem
                else:
                    raise KeyError(f"Label '{label_str}' from file {p} not in label2idx")
        lbl = self.label2idx[label_str]
        return torch.from_numpy(pts), torch.tensor(lbl, dtype=torch.long), p.stem

# -------------------------
# PointNet (as before, compact)
# -------------------------
def conv1d_bn(in_ch, out_ch):
    return nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, bias=False),
                         nn.BatchNorm1d(out_ch),
                         nn.ReLU(inplace=True))

class TNet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k=k
        self.conv1=conv1d_bn(k,64)
        self.conv2=conv1d_bn(64,128)
        self.conv3=conv1d_bn(128,1024)
        self.fc1=nn.Linear(1024,512); self.fc2=nn.Linear(512,256); self.fc3=nn.Linear(256,k*k)
        self.bn1=nn.BatchNorm1d(512); self.bn2=nn.BatchNorm1d(256)
        nn.init.zeros_(self.fc3.weight)
        with torch.no_grad():
            self.fc3.bias.copy_(torch.eye(k).view(-1))

    def forward(self,x):
        x=self.conv1(x); x=self.conv2(x); x=self.conv3(x)
        x=torch.max(x,2)[0]
        x=F.relu(self.bn1(self.fc1(x))); x=F.relu(self.bn2(self.fc2(x)))
        x=self.fc3(x); x=x.view(-1,self.k,self.k); return x

class PointNetCls(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        self.input_transform=TNet(k=3)
        self.conv1=conv1d_bn(3,64)
        self.feature_transform=TNet(k=64)
        self.conv2=conv1d_bn(64,128)
        self.conv3=conv1d_bn(128,1024)
        self.fc1=nn.Linear(1024,512); self.bn1=nn.BatchNorm1d(512)
        self.fc2=nn.Linear(512,256); self.bn2=nn.BatchNorm1d(256)
        self.dropout=nn.Dropout(dropout)
        self.fc3=nn.Linear(256,num_classes)

    def forward(self,x):
        x=x.transpose(2,1)
        trans=self.input_transform(x); x=torch.bmm(trans,x)
        x=self.conv1(x); trans_feat=self.feature_transform(x)
        x=x.transpose(2,1); x=torch.bmm(x, trans_feat); x=x.transpose(2,1)
        x=self.conv2(x); x=self.conv3(x)
        x=torch.max(x,2)
        x=torch.max(x,2)
        x=torch.max(x,2)
        x=torch.max(x,2)[0]
        x=torch.max(x,2)
        # simplified pooling:
        x = torch.max(x,2)[0]
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x, trans_feat

# -------------------------
# DGCNN (EdgeConv) - compact implementation
# -------------------------
def knn(x, k):
    # x: (B, C, N)
    inner = -2*torch.matmul(x.transpose(2,1), x)  # (B,N,N) with -2 x_i^T x_j
    xx = torch.sum(x**2, dim=1, keepdim=True)  # (B,1,N)
    pairwise = -xx.transpose(2,1) - inner - xx  # pairwise dist
    # topk largest of negative squared distances => smallest distances
    _, idx = pairwise.topk(k=k, dim=-1)  # (B,N,k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    # x: (B, C, N)
    B, C, N = x.size()
    if idx is None:
        idx = knn(x, k=k)  # (B, N, k)
    device = x.device

    idx_base = torch.arange(0, B, device=device).view(-1,1,1)*N
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2,1).contiguous()  # (B,N,C)
    feature = x.view(B*N, -1)[idx, :]
    feature = feature.view(B, N, k, C)
    x = x.view(B, N, 1, C).repeat(1,1,k,1)
    # edge feature = concat(neighbor - central, central)
    feat = torch.cat((feature - x, x), dim=3).permute(0,3,1,2).contiguous()  # (B,2C,N,k)
    return feat

class DGCNN(nn.Module):
    def __init__(self, num_classes=10, k=20, emb_dims=1024, dropout=0.5):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False), nn.BatchNorm1d(emb_dims), nn.LeakyReLU(0.2))
        self.linear1 = nn.Linear(emb_dims*2, 512, bias=False); self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(512, 256); self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (B, N, 3) -> make (B,3,N)
        x = x.transpose(2,1).contiguous()
        B, C, N = x.size()
        feat = get_graph_feature(x, k=self.k)  # (B,2C,N,k)
        x1 = self.conv1(feat); x1 = x1.max(dim=-1)[0]  # (B,64,N)
        feat = get_graph_feature(x1, k=self.k)
        x2 = self.conv2(feat); x2 = x2.max(dim=-1)[0]  # (B,64,N)
        feat = get_graph_feature(x2, k=self.k)
        x3 = self.conv3(feat); x3 = x3.max(dim=-1)[0]  # (B,128,N)
        feat = get_graph_feature(x3, k=self.k)
        x4 = self.conv4(feat); x4 = x4.max(dim=-1)[0]  # (B,256,N)
        x_concat = torch.cat((x1,x2,x3,x4), dim=1)  # (B,512,N)
        x = self.conv5(x_concat)  # (B,emb_dims,N)
        x1 = F.adaptive_max_pool1d(x, 1).view(B, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(B, -1)
        x = torch.cat((x1, x2), dim=1)  # (B, emb_dims*2)
        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x, None

# -------------------------
# Loss helpers
# -------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma; self.weight = weight; self.reduction=reduction
    def forward(self, inputs, targets):
        prob = F.softmax(inputs, dim=1)
        ce = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = ((1-pt)**self.gamma) * ce
        if self.reduction=='mean': return loss.mean()
        elif self.reduction=='sum': return loss.sum()
        else: return loss

# -------------------------
# IO helpers (same as before)
# -------------------------
def prepare_file_label_lists(data_dir: str, labels_csv: Optional[str] = None) -> Tuple[List[Path], List[str], List[str]]:
    root = Path(data_dir)
    files = sorted(list(root.glob("*.npy")))
    if not files:
        raise SystemExit(f"No .npy files found in {data_dir}")
    file_map = {_normalize_id(p.stem): p for p in files}

    if labels_csv and Path(labels_csv).exists():
        df = pd.read_csv(labels_csv)
        df = coerce_id_label_columns(df)
        if not {'id','label'}.issubset(df.columns):
            raise ValueError("labels_csv must contain 'id' and 'label'")
        df['id_norm'] = df['id'].astype(str).map(_normalize_id)
        matched_files=[]; matched_labels=[]; matched_ids=[]; missing=[]
        for _, row in df.iterrows():
            idn, lbl = row['id_norm'], str(row['label'])
            if idn in file_map:
                matched_files.append(file_map[idn]); matched_labels.append(lbl); matched_ids.append(row['id'])
            else:
                missing.append(row['id'])
        if missing:
            logger.warning(f"{len(missing)} ids from labels_csv not found among .npy files (examples {missing[:5]})")
        if len(matched_files)==0:
            raise SystemExit("After aligning with labels_csv no files remain.")
        return matched_files, matched_labels, matched_ids
    else:
        labels = [p.stem.split("_")[0] for p in files]
        ids = [p.stem for p in files]
        return files, labels, ids

# -------------------------
# Evaluation (TTA)
# -------------------------
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, tta: int =1):
    model.eval()
    all_true=[]; all_pred=[]; all_probs=[]
    with torch.no_grad():
        for batch in loader:
            if len(batch)==3:
                pts, labels, ids = batch
            else:
                pts, labels = batch
            pts = pts.to(device, dtype=torch.float32)
            probs_acc = None
            for t in range(max(1, tta)):
                if t==0:
                    inp = pts
                else:
                    inp_np = pts.cpu().numpy()
                    rng = np.random.RandomState(1234 + t)
                    for i in range(inp_np.shape[0]):
                        inp_np[i] = rotate_axis(inp_np[i], 'z', rng)
                        inp_np[i] = jitter(inp_np[i], rng, sigma=0.005, clip=0.02)
                    inp = torch.from_numpy(inp_np).to(device)
                out, _ = model(inp)
                prob = F.softmax(out, dim=1).cpu().numpy()
                probs_acc = prob if probs_acc is None else (probs_acc + prob)
            probs_acc = probs_acc / float(max(1, tta))
            preds = probs_acc.argmax(axis=1)
            all_true.extend(labels.cpu().numpy().tolist())
            all_pred.extend(preds.tolist())
            all_probs.extend(probs_acc.tolist())
    return np.asarray(all_true, dtype=int), np.asarray(all_pred, dtype=int), np.asarray(all_probs, dtype=float)

# -------------------------
# Training
# -------------------------
def train_pointnet(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out_dir / "train.log"); fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    files, labels, ids = prepare_file_label_lists(args.data_dir, args.labels_csv)
    n_samples = len(files)
    logger.info(f"{n_samples} samples, {len(set(labels))} classes found")

    classes = sorted(list(set(labels)))
    label2idx = {lbl: i for i,lbl in enumerate(classes)}
    with open(out_dir / "label2idx.json", "w", encoding="utf-8") as fhjson:
        json.dump(label2idx, fhjson, indent=2, ensure_ascii=False)

    # FAST subset option (keeps stratified)
    if args.fast:
        n_fast = min(args.fast_select, n_samples)
        if n_fast < n_samples:
            try:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=n_fast, random_state=args.seed)
                sel_idx, _ = next(sss.split(files, labels))
            except Exception:
                rng = np.random.RandomState(args.seed); sel_idx = rng.choice(n_samples, n_fast, replace=False)
            files = [files[i] for i in sel_idx]; labels = [labels[i] for i in sel_idx]
            logger.info(f"FAST mode: {len(files)} samples selected")

    # train/val split (stratified if possible)
    cnts = Counter(labels); min_count = min(cnts.values()) if cnts else 0
    if min_count >= 2:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=args.seed)
        tr_idx, va_idx = next(sss.split(files, labels)); logger.info("Using stratified split")
    else:
        ss = ShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=args.seed)
        tr_idx, va_idx = next(ss.split(files)); logger.warning("Falling back to random split")

    train_files = [files[i] for i in tr_idx]; train_labels = [labels[i] for i in tr_idx]
    val_files = [files[i] for i in va_idx]; val_labels = [labels[i] for i in va_idx]
    logger.info(f"Dataset sizes -> train: {len(train_files)}, val: {len(val_files)}")
    logger.info(f"Classes used: {classes}")

    # class weights mapping
    class_weights = None
    if args.use_class_weights:
        classes_train = np.array(sorted(list(set(train_labels))))
        cw = compute_class_weight("balanced", classes=classes_train, y=np.array(train_labels))
        weights = np.ones(len(classes), dtype=np.float32)
        for i,c in enumerate(classes):
            if c in classes_train:
                weights[i] = float(cw[list(classes_train).index(c)])
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        logger.info("Using class weights")

    # sampler
    sampler = None
    if args.use_sampler:
        idxs = [classes.index(l) for l in train_labels]
        counts = np.bincount(idxs, minlength=len(classes)).astype(np.float32)
        sample_weights = 1.0 / (counts[idxs] + 1e-12)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_files), replacement=True)
        logger.info("Using WeightedRandomSampler")

    # datasets & loaders
    train_ds = PointCloudDataset(train_files, label2idx, augment=True, npoints=args.npoints,
                                 seed=args.seed, augment_full_axes=args.augment_full_axes, normalize=args.normalize)
    val_ds   = PointCloudDataset(val_files,   label2idx, augment=False, npoints=args.npoints,
                                 seed=args.seed, normalize=args.normalize)
    num_workers = min(max(1, args.max_workers), (os.cpu_count() or 1))
    pin_memory = (device.type == "cuda")
    if sampler is not None:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # model selection
    if args.arch == 'dgcnn':
        model = DGCNN(num_classes=len(classes), k=args.dgcnn_k, emb_dims=args.dgcnn_emb, dropout=args.dropout).to(device)
    else:
        model = PointNetCls(num_classes=len(classes), dropout=args.dropout).to(device)

    # loss
    if args.use_focal:
        criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights)
    else:
        if args.label_smoothing > 0.0 and hasattr(nn, "CrossEntropyLoss"):
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # scheduler: OneCycle or ReduceLROnPlateau
    use_onecycle = args.use_onecycle
    if use_onecycle and sampler is not None:
        logger.warning("OneCycleLR + sampler not recommended -> disabling OneCycle.")
        use_onecycle = False
    if use_onecycle:
        steps_per_epoch = max(1, len(train_loader))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch)
        logger.info("Using OneCycleLR")
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
        logger.info("Using ReduceLROnPlateau")

    # AMP
    use_amp = torch.cuda.is_available()
    if use_amp:
        try:
            scaler = torch.amp.GradScaler()
        except Exception:
            scaler = torch.cuda.amp.GradScaler()
        try:
            autocast_ctx = lambda: torch.amp.autocast(device_type=device.type)
        except Exception:
            autocast_ctx = lambda: torch.cuda.amp.autocast()
    else:
        scaler = None
        autocast_ctx = lambda: nullcontext()

    # EMA
    ema = None
    if args.use_ema:
        class ModelEMA:
            def __init__(self, model, decay=0.999):
                self.decay = decay
                self.ema = type(model)(num_classes=len(classes)) if isinstance(model, PointNetCls) else DGCNN(num_classes=len(classes), k=args.dgcnn_k, emb_dims=args.dgcnn_emb, dropout=args.dropout)
                # if creation fails fallback to same stateful copy
                try:
                    self.ema.load_state_dict(model.state_dict())
                except Exception:
                    self.ema = type(model)(num_classes=len(classes)).to(device); self.ema.load_state_dict(model.state_dict())
                self.ema.to(device); self.ema.eval()
                for p in self.ema.parameters(): p.requires_grad_(False)
            @torch.no_grad()
            def update(self, model):
                msd = model.state_dict()
                for k, v in self.ema.state_dict().items():
                    v.copy_(v*self.decay + (1.0-self.decay)*msd[k].detach())
            def state_dict(self): return self.ema.state_dict()
        ema = ModelEMA(model, decay=args.ema_decay)
        logger.info("Using EMA")

    best_val = 0.0; patience_counter = 0; history=[]

    enable_early_stop = (not args.no_early_stop) and (args.patience is not None)
    patience = args.patience if args.patience is not None else 9999999

    # training loop
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0; total = 0; correct = 0

        for batch_idx, batch in enumerate(train_loader):
            if len(batch)==3:
                pts, labels_batch, _ = batch
            else:
                pts, labels_batch = batch
            pts = pts.to(device, dtype=torch.float32); labels_batch = labels_batch.to(device)

            # mixup
            use_mix = False
            if args.mixup_alpha > 0 and random.random() < args.mixup_prob:
                perm = torch.randperm(pts.size(0), device=pts.device)
                pts2 = pts[perm]; labels2 = labels_batch[perm]
                lam = np.random.beta(args.mixup_alpha, args.mixup_alpha); lam_t = float(lam)
                pts_input = lam_t * pts + (1.0 - lam_t) * pts2; use_mix = True
            else:
                pts_input = pts; labels2 = None; lam_t = 1.0

            optimizer.zero_grad()
            with autocast_ctx():
                outputs, trans_feat = model(pts_input)
                if use_mix:
                    loss1 = criterion(outputs, labels_batch); loss2 = criterion(outputs, labels2)
                    loss = lam_t * loss1 + (1.0 - lam_t) * loss2
                else:
                    loss = criterion(outputs, labels_batch)
            if args.feat_reg_weight > 0 and trans_feat is not None:
                loss = loss + args.feat_reg_weight * 0.0  # for DGCNN trans_feat None; for PointNet keep original reg if needed

            if scaler is not None:
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()

            # scheduler step after optimizer when OneCycle
            if use_onecycle:
                try: scheduler.step()
                except Exception: pass

            preds = outputs.argmax(dim=1)
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)
            total_loss += float(loss.item()) * labels_batch.size(0)

            if ema is not None:
                ema.update(model)

        train_loss = total_loss / total if total>0 else 0.0
        train_acc = correct / total if total>0 else 0.0

        # evaluate
        eval_model = ema.ema if ema is not None else model
        y_true, y_pred, y_probs = evaluate_model(eval_model, val_loader, device, tta=args.tta)

        # val NLL
        val_loss = 0.0
        if y_probs is not None and y_probs.size != 0 and y_probs.ndim == 2:
            probs = y_probs; y_true_idx = np.asarray(y_true, dtype=int)
            n = min(probs.shape[0], len(y_true_idx))
            probs_used = probs[:n]; y_true_idx = y_true_idx[:n]
            probs_clipped = np.clip(probs_used, 1e-12, 1.0)
            per_sample_nll = -np.log(probs_clipped[np.arange(n), y_true_idx])
            val_loss = float(np.mean(per_sample_nll)) if per_sample_nll.size else 0.0

        val_acc = float((y_true == y_pred).mean()) if len(y_true) > 0 else 0.0

        if (not use_onecycle) and (scheduler is not None):
            try: scheduler.step(val_loss)
            except Exception: pass

        history.append((epoch, train_loss, train_acc, val_loss, val_acc))
        if (args.log_every and epoch % args.log_every==0) or args.verbose or epoch==args.epochs:
            logger.info(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        else:
            logger.debug(f"Epoch {epoch}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

        if args.save_every and (epoch % args.save_every == 0 or epoch==args.epochs):
            try:
                np.save(out_dir / f"val_probas_epoch{epoch}.npy", y_probs)
                pd.DataFrame({"id":[p.stem for p in val_files],
                              "true":[classes[int(t)] for t in y_true],
                              "pred":[classes[int(p)] for p in y_pred]}).to_csv(out_dir / f"val_preds_epoch{epoch}.csv", index=False)
            except Exception as e:
                logger.debug(f"Could not save val outputs epoch {epoch}: {e}")

        # checkpoint best
        if val_acc > best_val:
            best_val = val_acc
            ckpt = {"epoch":epoch, "model_state_dict":model.state_dict(), "optimizer_state_dict":optimizer.state_dict(),
                    "label2idx":label2idx, "args":vars(args)}
            if ema is not None: ckpt["ema_state_dict"] = ema.state_dict()
            torch.save(ckpt, out_dir / "checkpoint_full.pth")
            torch.save(model.state_dict(), out_dir / "checkpoint.pth")
            logger.info(f"  -> New best model saved (val_acc={best_val:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        if enable_early_stop and patience_counter >= patience:
            logger.info("Early stopping triggered."); break

    # final saves
    torch.save(model.state_dict(), out_dir / "last_checkpoint.pth")
    pd.DataFrame(history, columns=["epoch","train_loss","train_acc","val_loss","val_acc"]).to_csv(out_dir / "metrics.csv", index=False)
    with open(out_dir / "label2idx.json", "w", encoding="utf-8") as fh:
        json.dump(label2idx, fh, indent=2, ensure_ascii=False)

    try:
        ckpt = torch.load(out_dir / "checkpoint_full.pth", map_location="cpu")
        state = ckpt.get("ema_state_dict", None) or ckpt.get("model_state_dict")
        model.load_state_dict(state); model.to(device)
        y_true, y_pred, y_probs = evaluate_model(model, val_loader, device, tta=args.tta)
        report = classification_report([classes[int(t)] for t in y_true], [classes[int(p)] for p in y_pred], zero_division=0)
        logger.info("Final validation classification report:\n" + report)
        cm = confusion_matrix([classes[int(t)] for t in y_true], [classes[int(p)] for p in y_pred], labels=classes)
        (out_dir / "val_classification_report.txt").write_text(report)
        np.save(out_dir / "val_confusion_matrix.npy", cm)
    except Exception as e:
        logger.warning(f"Failed to compute final report: {e}")

    logger.info(f"Training finished. Best val_acc={best_val:.4f}. Artifacts saved to {out_dir}")

# -------------------------
# CLI
# -------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Train PointNet / DGCNN improved")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--labels_csv", type=str, default=None)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--arch", type=str, default="dgcnn", choices=["pointnet","dgcnn"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--npoints", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--max_lr", type=float, default=1e-2)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--no_early_stop", action="store_true")
    p.add_argument("--fast", action="store_true"); p.add_argument("--fast_select", type=int, default=200)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--use_sampler", action="store_true")
    p.add_argument("--use_class_weights", action="store_true")
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--use_onecycle", action="store_true")
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--grad_clip", type=float, default=2.0)
    p.add_argument("--tta", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--max_workers", type=int, default=8)
    p.add_argument("--feat_reg_weight", type=float, default=0.0)
    p.add_argument("--mixup_alpha", type=float, default=0.0)
    p.add_argument("--mixup_prob", type=float, default=0.0)
    p.add_argument("--augment_full_axes", action="store_true")
    p.add_argument("--use_focal", action="store_true")
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--dgcnn_k", type=int, default=20)
    p.add_argument("--dgcnn_emb", type=int, default=1024)
    return p

def main(argv=None):
    args = build_argparser().parse_args(argv)
    if args.verbose: logger.setLevel(logging.DEBUG)
    train_pointnet(args)

if __name__ == "__main__":
    main()
