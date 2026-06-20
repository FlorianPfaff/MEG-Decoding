#!/usr/bin/env python3
"""Compact PyTorch baselines for strict no-calibration MEG LOSO decoding.

Models: EEGNet, LF-CNN-style, VAR-CNN-style, and HGRN-style.
Inputs are trial tensors shaped [trials, channels, time] with labels and subject IDs.
"""
from __future__ import annotations

import argparse, csv, json, math, random, tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------- models -----------------------------
class ConvBlock1d(nn.Module):
    def __init__(self, cin, cout, k, *, groups=1, dilation=1, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, padding=dilation * (k // 2), dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm1d(cout), nn.GELU(), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class EEGNet(nn.Module):
    def __init__(self, n_channels, n_classes, *, f1=8, depth_multiplier=2, temporal_kernel=64, sep_kernel=16, dropout=0.25):
        super().__init__(); fsp = f1 * depth_multiplier
        self.net = nn.Sequential(
            nn.Unflatten(1, (1, n_channels)),
            nn.Conv2d(1, f1, (1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False), nn.BatchNorm2d(f1),
            nn.Conv2d(f1, fsp, (n_channels, 1), groups=f1, bias=False), nn.BatchNorm2d(fsp), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
            nn.Conv2d(fsp, fsp, (1, sep_kernel), padding=(0, sep_kernel // 2), groups=fsp, bias=False),
            nn.Conv2d(fsp, fsp, (1, 1), bias=False), nn.BatchNorm2d(fsp), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(fsp, n_classes))
    def forward(self, x): return self.net(x)

class LFCNN(nn.Module):
    """LF-CNN-style: learned spatial latent factors, then depthwise temporal filters."""
    def __init__(self, n_channels, n_classes, *, hidden=48, temporal_kernel=31, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, hidden, 1, bias=False), nn.BatchNorm1d(hidden), nn.GELU(),
            ConvBlock1d(hidden, hidden, temporal_kernel, groups=hidden, dropout=dropout),
            nn.Conv1d(hidden, hidden, 1, bias=False), nn.BatchNorm1d(hidden), nn.GELU(), nn.AvgPool1d(4, ceil_mode=True),
            ConvBlock1d(hidden, hidden, 15, groups=hidden, dropout=dropout),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))
    def forward(self, x): return self.net(x)

class ResTemporal(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__(); self.block = nn.Sequential(ConvBlock1d(channels, channels, 15, dilation=dilation, dropout=dropout), nn.Conv1d(channels, channels, 1, bias=False), nn.BatchNorm1d(channels)); self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.block(x))

class VARCNN(nn.Module):
    """VAR-CNN-style: depthwise lag filters followed by dilated temporal blocks."""
    def __init__(self, n_channels, n_classes, *, hidden=64, lag_order=8, n_blocks=3, dropout=0.25):
        super().__init__(); k = lag_order + 1
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, n_channels, k, padding=k // 2, groups=n_channels, bias=False), nn.BatchNorm1d(n_channels), nn.GELU(),
            nn.Conv1d(n_channels, hidden, 1, bias=False), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
            *[ResTemporal(hidden, 2 ** i, dropout) for i in range(n_blocks)],
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))
    def forward(self, x): return self.net(x)

class HGRN(nn.Module):
    """HGRN-style: spatial projection, GRU temporal model, attention pooling."""
    def __init__(self, n_channels, n_classes, *, hidden=64, bidirectional=True, dropout=0.25):
        super().__init__(); self.spatial = nn.Sequential(nn.Conv1d(n_channels, hidden, 1, bias=False), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=bidirectional); d = hidden * (2 if bidirectional else 1)
        self.att = nn.Sequential(nn.Linear(d, max(1, d // 2)), nn.Tanh(), nn.Linear(max(1, d // 2), 1)); self.cls = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, n_classes))
    def forward(self, x):
        z = self.spatial(x).transpose(1, 2); z, _ = self.gru(z); w = torch.softmax(self.att(z), dim=1); return self.cls((w * z).sum(1))

def make_model(name, n_channels, n_classes, dropout=0.25):
    table = {"eegnet": EEGNet, "lfcnn": LFCNN, "lf-cnn": LFCNN, "varcnn": VARCNN, "var-cnn": VARCNN, "hgrn": HGRN}
    if name.lower() not in table: raise ValueError(f"Unknown model {name}; choose {sorted(table)}")
    return table[name.lower()](n_channels, n_classes, dropout=dropout)


# ----------------------------- data + metrics -----------------------------
@dataclass
class Dataset:
    x: np.ndarray; y: np.ndarray; subjects: np.ndarray; label_values: np.ndarray | None = None
    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=np.float32); self.y = np.asarray(self.y).reshape(-1); self.subjects = np.asarray(self.subjects).reshape(-1)
        if self.x.ndim != 3: raise ValueError(f"X must be [trials, channels, time], got {self.x.shape}")
        if len(self.y) != len(self.x) or len(self.subjects) != len(self.x): raise ValueError("X, y, subjects length mismatch")
    @property
    def n_channels(self): return int(self.x.shape[1])
    @property
    def n_classes(self): return int(np.unique(self.y).size)

def load_dataset(path: Path, drop_nonpositive=True) -> Dataset:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as f:
            x = f["X"] if "X" in f.files else f["x"]; y = f["y"]; s = f["subjects"]
    else:
        from scipy.io import loadmat
        m = loadmat(path, squeeze_me=True); x = m["X"]; y = m["y"]; s = m["subjects"]
    x = np.asarray(x, dtype=np.float32); y = np.asarray(y).reshape(-1); s = np.asarray(s).reshape(-1)
    if x.shape[0] != y.shape[0] and x.shape[-1] == y.shape[0]: x = np.moveaxis(x, -1, 0)  # tolerate MATLAB [C,T,N]
    if drop_nonpositive:
        keep = y > 0; x, y, s = x[keep], y[keep], s[keep]
    vals = np.array(sorted(np.unique(y))); enc = np.array([{v: i for i, v in enumerate(vals)}[v] for v in y], dtype=np.int64)
    return Dataset(x, enc, s, vals)

def fit_standardizer(x):
    mean = x.mean(axis=(0, 2), keepdims=True); std = x.std(axis=(0, 2), keepdims=True); return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)
def standardize(x, mean, std): return ((x - mean) / std).astype(np.float32)
def bal_acc(y, pred, n_classes):
    rec = [np.mean(pred[y == c] == c) for c in range(n_classes) if np.any(y == c)]; return float(np.mean(rec))
def split_masks(subjects, target, n_val_subjects, seed):
    subjects = np.asarray(subjects); target = int(target) if str(target).isdigit() else target; test = subjects == target
    source_subjects = np.array(sorted(set(subjects.tolist()) - {target}))
    rng = np.random.default_rng(seed); n_val = min(max(1, n_val_subjects), max(1, len(source_subjects) - 1)); val_sub = rng.choice(source_subjects, n_val, replace=False)
    val = np.isin(subjects, val_sub); train = (~test) & (~val)
    if not np.any(train) or not np.any(test): raise ValueError(f"Invalid split for target={target}")
    return train, val, test


# ----------------------------- training -----------------------------
@dataclass
class FoldResult:
    model: str; target_subject: str; accuracy: float; balanced_accuracy: float; chance: float; n_train: int; n_val: int; n_test: int; best_epoch: int; best_val_accuracy: float

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def loader(x, y, batch, shuffle): return DataLoader(TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long()), batch_size=batch, shuffle=shuffle)

def evaluate(model, dl, device, n_classes):
    model.eval(); ys = []; ps = []
    with torch.no_grad():
        for xb, yb in dl:
            p = model(xb.to(device)).argmax(1).cpu().numpy(); ps.append(p); ys.append(yb.numpy())
    y = np.concatenate(ys); p = np.concatenate(ps); return float(np.mean(y == p)), bal_acc(y, p, n_classes)

def train_fold(ds, model_name, target, args, device):
    train_m, val_m, test_m = split_masks(ds.subjects, target, args.n_val_subjects, args.seed)
    mean, std = fit_standardizer(ds.x[train_m])
    xtr, xv, xte = standardize(ds.x[train_m], mean, std), standardize(ds.x[val_m], mean, std), standardize(ds.x[test_m], mean, std)
    ytr, yv, yte = ds.y[train_m].astype(np.int64), ds.y[val_m].astype(np.int64), ds.y[test_m].astype(np.int64)
    tr_dl, va_dl, te_dl = loader(xtr, ytr, args.batch_size, True), loader(xv, yv, args.batch_size, False), loader(xte, yte, args.batch_size, False)
    model = make_model(model_name, ds.n_channels, ds.n_classes, dropout=args.dropout).to(device)
    counts = np.bincount(ytr, minlength=ds.n_classes).astype(np.float32); w = counts.sum() / np.maximum(counts, 1) / ds.n_classes
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device) if not args.no_class_weighting else None)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, args.epochs))
    best, best_epoch, best_val, bad = None, 0, -math.inf, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in tr_dl:
            opt.zero_grad(set_to_none=True); loss = criterion(model(xb.to(device)), yb.to(device)); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sched.step(); va, _ = evaluate(model, va_dl, device, ds.n_classes)
        if va > best_val: best_val, best_epoch, best, bad = va, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else: bad += 1
        if bad >= args.patience: break
    if best is not None: model.load_state_dict(best)
    acc, bacc = evaluate(model, te_dl, device, ds.n_classes)
    return FoldResult(model_name, str(target), acc, bacc, 1 / ds.n_classes, int(train_m.sum()), int(val_m.sum()), int(test_m.sum()), best_epoch, float(best_val))

def write_results(rows, prefix: Path):
    prefix.parent.mkdir(parents=True, exist_ok=True); dicts = [asdict(r) for r in rows]
    with prefix.with_suffix(".csv").open("w", newline="") as f: w = csv.DictWriter(f, fieldnames=dicts[0].keys()); w.writeheader(); w.writerows(dicts)
    summary = {}
    for m in sorted({r.model for r in rows}):
        rs = [r for r in rows if r.model == m]; summary[m] = {"mean_accuracy": float(np.mean([r.accuracy for r in rs])), "std_accuracy": float(np.std([r.accuracy for r in rs])), "mean_balanced_accuracy": float(np.mean([r.balanced_accuracy for r in rs])), "chance": rs[0].chance, "n_folds": len(rs)}
    with prefix.with_suffix(".json").open("w") as f: json.dump({"folds": dicts, "summary": summary}, f, indent=2)
    print(json.dumps(summary, indent=2)); print(f"Wrote {prefix.with_suffix('.csv')} and {prefix.with_suffix('.json')}")

def run_smoke_test():
    torch.set_num_threads(1); rng = np.random.default_rng(7); nsub, ncls, ntr, nch, nt = 3, 4, 4, 8, 48
    templates = rng.normal(size=(ncls, nch, nt)).astype(np.float32) * 0.15; xs=[]; ys=[]; ss=[]
    for s in range(1, nsub + 1):
        shift = rng.normal(size=(nch, 1)).astype(np.float32) * 0.05
        for c in range(ncls):
            for _ in range(ntr): xs.append(templates[c] + shift + rng.normal(size=(nch, nt)).astype(np.float32) * 0.7); ys.append(c + 1); ss.append(s)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "synthetic.npz"; np.savez(p, X=np.stack(xs), y=np.array(ys), subjects=np.array(ss))
        main(["--data", str(p), "--models", "eegnet", "lfcnn", "varcnn", "hgrn", "--target-subjects", "1", "--epochs", "1", "--patience", "1", "--batch-size", "8", "--device", "cpu", "--output-prefix", str(Path(tmp) / "smoke")])

def parse(argv=None):
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", type=Path, help=".npz or .mat tensor file with X, y, subjects")
    ap.add_argument("--models", nargs="+", default=["eegnet", "lfcnn", "varcnn", "hgrn"])
    ap.add_argument("--target-subjects", nargs="*", default=None)
    ap.add_argument("--n-val-subjects", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=80); ap.add_argument("--patience", type=int, default=15); ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--weight-decay", type=float, default=1e-3); ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--no-class-weighting", action="store_true"); ap.add_argument("--include-zero-label", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--output-prefix", type=Path, default=Path("results/pytorch_loso")); ap.add_argument("--smoke-test", action="store_true")
    a = ap.parse_args(argv)
    if not a.smoke_test and a.data is None: ap.error("--data is required unless --smoke-test is used")
    return a

def main(argv=None):
    args = parse(argv)
    if args.smoke_test: return run_smoke_test()
    seed_all(args.seed); ds = load_dataset(args.data, drop_nonpositive=not args.include_zero_label); device = torch.device(args.device)
    targets = args.target_subjects or [str(s) for s in sorted(np.unique(ds.subjects).tolist())]
    results = []
    for m in args.models:
        for t in targets:
            print(f"Training {m}, held-out subject {t}"); r = train_fold(ds, m, t, args, device); print(r); results.append(r)
    write_results(results, args.output_prefix)

if __name__ == "__main__": main()
