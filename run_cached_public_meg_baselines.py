#!/usr/bin/env python3
"""Run the PyTorch MEG baselines on one cached public dataset.

The self-hosted runner is expected to have datasets under
/home/github-runner/.cache/datasets. This script searches for a compatible
public MEG representation and converts a small, bounded subset to the
X/y/subjects tensor expected by pytorch_meg_baselines.py.

Supported inputs, in priority order:
  1. .npz/.mat files that already contain X, y, subjects arrays.
  2. MNE Epochs FIF files (*-epo.fif, *-epo.fif.gz, *epo*.fif*).
  3. MNE Raw FIF files with stimulus events, e.g. MNE visual_92_categories.

The goal is a quick strict-LOSO smoke benchmark on real public data, not a final
hyperparameter-optimized result.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

TENSOR_SUFFIXES = {".npz", ".mat"}
EPOCH_PATTERNS = ("*-epo.fif", "*-epo.fif.gz", "*epo*.fif", "*epo*.fif.gz")
RAW_SKIP_WORDS = (
    "epo", "epoch", "ave", "cov", "fwd", "inv", "trans", "src", "bem",
    "head", "fid", "annot", "emptyroom", "erm", "sss_info",
)


def log(msg: str) -> None:
    print(msg, flush=True)


def is_sidecar_or_hidden_data_file(path: Path) -> bool:
    """Return True for macOS AppleDouble sidecars and other hidden data files."""
    return path.name.startswith("._") or path.name.startswith(".")


def iter_files(root: Path, max_files: int = 50000) -> Iterable[Path]:
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        # Keep traversal bounded and avoid common non-data folders.
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", "node_modules"}]
        for filename in filenames:
            path = Path(dirpath) / filename
            if is_sidecar_or_hidden_data_file(path):
                continue
            count += 1
            if count > max_files:
                return
            yield path


def combined_suffix(path: Path) -> str:
    return "".join(path.suffixes[-2:]) if path.name.endswith(".fif.gz") else path.suffix


def inventory(root: Path, max_entries: int = 200) -> dict:
    suffix_counts = Counter()
    examples: list[str] = []
    for p in iter_files(root):
        suffix = combined_suffix(p)
        suffix_counts[suffix or "<none>"] += 1
        if len(examples) < max_entries and (
            suffix in TENSOR_SUFFIXES or "epo" in p.name.lower() or p.suffix in {".fif", ".gz", ".npy", ".npz", ".mat"}
        ):
            examples.append(str(p))
    return {"root": str(root), "suffix_counts": dict(suffix_counts.most_common(50)), "examples": examples}


def try_load_tensor_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as f:
                keys = set(f.files)
                x_key = "X" if "X" in keys else "x" if "x" in keys else None
                y_key = "y" if "y" in keys else "labels" if "labels" in keys else None
                s_key = "subjects" if "subjects" in keys else "subject" if "subject" in keys else None
                if not (x_key and y_key and s_key):
                    return None
                x, y, subjects = f[x_key], f[y_key], f[s_key]
        elif path.suffix == ".mat":
            try:
                from scipy.io import loadmat
                m = loadmat(path, squeeze_me=True)
                if not all(k in m for k in ("X", "y", "subjects")):
                    return None
                x, y, subjects = m["X"], m["y"], m["subjects"]
            except NotImplementedError:
                # Likely HDF5/v7.3 MAT. Keep this conservative because array
                # orientation and structs are often dataset-specific.
                return None
        else:
            return None
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y).reshape(-1)
        subjects = np.asarray(subjects).reshape(-1)
        if x.ndim != 3:
            return None
        if x.shape[0] != len(y) and x.shape[-1] == len(y):
            x = np.moveaxis(x, -1, 0)
        if x.shape[0] == len(y) == len(subjects):
            return x, y, subjects
    except Exception as exc:  # noqa: BLE001 - inventory/probing should continue.
        log(f"Skipping tensor candidate {path}: {exc}")
    return None


def subject_from_path(path: Path) -> str:
    """Extract a stable subject identifier from common MEG dataset filenames.

    The ordering matters. We must match full words such as ``sample_subject_0``
    before the BIDS-style ``sub`` pattern, otherwise ``subject`` can be parsed as
    ``sub-ject``.
    """
    text = str(path)
    patterns = [
        r"sample[_-]subject[_-]?([A-Za-z0-9]+)",
        r"subject[_-]?([A-Za-z0-9]+)",
        r"participant[_-]?([A-Za-z0-9]+)",
        r"(?<![A-Za-z0-9])sub[_-]?([A-Za-z0-9]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.I)
        if matches:
            value = matches[-1]
            if isinstance(value, tuple):
                value = next((v for v in value if v), "")
            if value:
                return f"sub-{value}"

    # Fall back to the nearest directory name to avoid making each file a new subject.
    for parent in path.parents:
        name = parent.name
        lower = name.lower()
        if lower.startswith(("sub-", "sub_", "subject", "participant")):
            return name
    return path.parent.name


def find_mne_epoch_files(root: Path, limit: int = 200) -> list[Path]:
    found: list[Path] = []
    for pattern in EPOCH_PATTERNS:
        found.extend(p for p in root.rglob(pattern) if not is_sidecar_or_hidden_data_file(p))
    # Deduplicate while preserving deterministic order.
    unique = sorted(set(found), key=lambda p: str(p))
    return unique[:limit]


def find_mne_raw_files(root: Path, limit: int = 200) -> list[Path]:
    candidates: list[Path] = []
    for p in iter_files(root):
        suffix = combined_suffix(p)
        if suffix not in {".fif", ".fif.gz"}:
            continue
        name = p.name.lower()
        if any(word in name for word in RAW_SKIP_WORDS):
            continue
        # Avoid separately opening split-file continuations when the first file
        # points to them internally.
        if re.search(r"-\d+\.fif(\.gz)?$", name):
            continue
        candidates.append(p)
    # Prioritize likely data files such as raw/tsss/mc files.
    candidates = sorted(
        set(candidates),
        key=lambda p: (
            not any(k in p.name.lower() for k in ("raw", "tsss", "sss", "meg")),
            str(p),
        ),
    )
    return candidates[:limit]


def import_mne():
    try:
        import mne
        return mne
    except Exception as exc:  # noqa: BLE001
        log(f"MNE is not importable: {exc}")
        return None


def load_mne_epochs(
    paths: list[Path],
    *,
    max_subjects: int,
    tmin: float | None,
    tmax: float | None,
    resample_hz: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict] | None:
    mne = import_mne()
    if mne is None:
        return None

    by_subject: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        by_subject[subject_from_path(p)].append(p)
    selected_subjects = sorted(by_subject)[:max_subjects]
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ss: list[np.ndarray] = []
    used_files: list[str] = []

    for subj in selected_subjects:
        for p in by_subject[subj]:
            try:
                epochs = mne.read_epochs(p, preload=True, verbose="ERROR")
                if tmin is not None or tmax is not None:
                    lo = epochs.times[0] if tmin is None else max(float(tmin), float(epochs.times[0]))
                    hi = epochs.times[-1] if tmax is None else min(float(tmax), float(epochs.times[-1]))
                    if lo < hi:
                        epochs.crop(tmin=lo, tmax=hi)
                if resample_hz is not None and epochs.info["sfreq"] > resample_hz:
                    epochs.resample(resample_hz, verbose="ERROR")
                data = epochs.get_data(picks="meg").astype(np.float32)
                labels = epochs.events[:, 2]
                if data.ndim != 3 or len(labels) != len(data) or len(np.unique(labels)) < 2:
                    continue
                xs.append(data)
                ys.append(labels)
                ss.append(np.array([subj] * len(labels), dtype=object))
                used_files.append(str(p))
            except Exception as exc:  # noqa: BLE001
                log(f"Skipping epochs candidate {p}: {exc}")
    if not xs:
        return None
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    subjects = np.concatenate(ss, axis=0)
    meta = {"kind": "mne_epochs", "used_files": used_files, "selected_subjects": selected_subjects}
    return x, y, subjects, meta


def find_events_robust(mne, raw):
    # Let MNE choose the stim channel first; fall back to common Neuromag names.
    errors = []
    for stim_channel in (None, "STI101", "STI 014", "STI102"):
        try:
            return mne.find_events(raw, stim_channel=stim_channel, shortest_event=1, verbose="ERROR")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{stim_channel}: {exc}")
    log("Could not find events. Tried: " + " | ".join(errors))
    return None


def bounded_events(events: np.ndarray, *, max_classes: int, max_trials_per_class: int, event_code_max: int | None, seed: int) -> np.ndarray:
    events = events[events[:, 2] > 0]
    if event_code_max is not None:
        events = events[events[:, 2] <= event_code_max]
    if len(events) == 0:
        return events
    rng = np.random.default_rng(seed)
    selected_codes = sorted(np.unique(events[:, 2]).tolist())[:max_classes]
    selected_rows: list[int] = []
    for code in selected_codes:
        idx = np.where(events[:, 2] == code)[0]
        if len(idx) > max_trials_per_class:
            idx = rng.choice(idx, size=max_trials_per_class, replace=False)
        selected_rows.extend(idx.tolist())
    selected_rows = sorted(selected_rows)
    return events[selected_rows]


def load_mne_raw_events(
    paths: list[Path],
    *,
    max_subjects: int,
    max_classes: int,
    max_trials_per_subject_class: int,
    event_code_max: int | None,
    tmin: float,
    tmax: float,
    resample_hz: float | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict] | None:
    mne = import_mne()
    if mne is None:
        return None

    by_subject: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        by_subject[subject_from_path(p)].append(p)
    selected_subjects = sorted(by_subject)[:max_subjects]
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ss: list[np.ndarray] = []
    used_files: list[str] = []

    for subj in selected_subjects:
        for p in by_subject[subj]:
            try:
                raw = mne.io.read_raw_fif(p, preload=False, on_split_missing="warn", verbose="ERROR")
                events = find_events_robust(mne, raw)
                if events is None:
                    continue
                events = bounded_events(
                    events,
                    max_classes=max_classes,
                    max_trials_per_class=max_trials_per_subject_class,
                    event_code_max=event_code_max,
                    seed=seed,
                )
                if len(events) == 0 or len(np.unique(events[:, 2])) < 2:
                    log(f"Skipping raw candidate {p}: not enough usable event classes after bounding")
                    continue
                event_id = {str(code): int(code) for code in sorted(np.unique(events[:, 2]).tolist())}
                epochs = mne.Epochs(
                    raw,
                    events,
                    event_id=event_id,
                    tmin=tmin,
                    tmax=tmax,
                    baseline=None,
                    preload=True,
                    reject_by_annotation=False,
                    picks="meg",
                    verbose="ERROR",
                )
                if resample_hz is not None and epochs.info["sfreq"] > resample_hz:
                    epochs.resample(resample_hz, verbose="ERROR")
                data = epochs.get_data().astype(np.float32)
                labels = epochs.events[:, 2]
                if data.ndim != 3 or len(labels) != len(data) or len(np.unique(labels)) < 2:
                    continue
                xs.append(data)
                ys.append(labels)
                ss.append(np.array([subj] * len(labels), dtype=object))
                used_files.append(str(p))
            except Exception as exc:  # noqa: BLE001
                log(f"Skipping raw candidate {p}: {exc}")
    if not xs:
        return None
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    subjects = np.concatenate(ss, axis=0)
    meta = {"kind": "mne_raw_events", "used_files": used_files, "selected_subjects": selected_subjects}
    return x, y, subjects, meta


def restrict_dataset(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    *,
    max_subjects: int,
    max_classes: int,
    max_trials_per_subject_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    subjects = subjects.astype(str)

    # Remove nonpositive numeric labels if possible; event code 0 is often non-class.
    try:
        numeric_y = y.astype(float)
        keep = numeric_y > 0
        x, y, subjects = x[keep], y[keep], subjects[keep]
    except Exception:
        pass

    class_counts = Counter(y.tolist())
    selected_classes = sorted(class_counts.keys(), key=lambda v: str(v))[:max_classes]
    keep_class = np.isin(y, selected_classes)
    x, y, subjects = x[keep_class], y[keep_class], subjects[keep_class]

    subject_counts = Counter(subjects.tolist())
    selected_subjects = [s for s, _ in subject_counts.most_common(max_subjects)]
    keep_subject = np.isin(subjects, selected_subjects)
    x, y, subjects = x[keep_subject], y[keep_subject], subjects[keep_subject]

    selected_indices: list[int] = []
    for subj in sorted(set(subjects.tolist())):
        for cls in selected_classes:
            idx = np.where((subjects == subj) & (y == cls))[0]
            if len(idx) == 0:
                continue
            if len(idx) > max_trials_per_subject_class:
                idx = rng.choice(idx, size=max_trials_per_subject_class, replace=False)
            selected_indices.extend(idx.tolist())
    selected_indices = sorted(selected_indices)
    x, y, subjects = x[selected_indices], y[selected_indices], subjects[selected_indices]

    label_values = np.array(sorted(set(y.tolist()), key=lambda v: str(v)))
    label_map = {v: i + 1 for i, v in enumerate(label_values)}
    y_encoded = np.array([label_map[v] for v in y], dtype=np.int64)
    meta = {
        "n_trials": int(len(y_encoded)),
        "n_channels": int(x.shape[1]),
        "n_times": int(x.shape[2]),
        "n_subjects": int(len(set(subjects.tolist()))),
        "n_classes": int(len(label_values)),
        "chance": float(1.0 / max(1, len(label_values))),
        "selected_subjects": sorted(set(subjects.tolist())),
        "label_values": [str(v) for v in label_values.tolist()],
        "class_counts": {str(k): int(v) for k, v in Counter(y_encoded.tolist()).items()},
        "subject_counts": {str(k): int(v) for k, v in Counter(subjects.tolist()).items()},
    }
    if meta["n_subjects"] < 2 or meta["n_classes"] < 2:
        raise RuntimeError(f"Need at least 2 subjects and 2 classes after restriction, got {meta}")
    return x.astype(np.float32), y_encoded, subjects.astype(str), meta


def save_subset(args, x: np.ndarray, y: np.ndarray, subjects: np.ndarray, meta: dict) -> tuple[Path, dict]:
    out = args.output_dir / "cached_public_meg_subset.npz"
    np.savez_compressed(out, X=x, y=y, subjects=subjects)
    (args.output_dir / "dataset_info.json").write_text(json.dumps(meta, indent=2))
    return out, meta


def find_dataset(args) -> tuple[Path, dict]:
    root = args.cache_root
    inv = inventory(root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cache_inventory.json").write_text(json.dumps(inv, indent=2))
    log("Cache inventory written to " + str(args.output_dir / "cache_inventory.json"))
    log(json.dumps(inv["suffix_counts"], indent=2)[:4000])

    tensor_candidates = [p for p in iter_files(root) if p.suffix in TENSOR_SUFFIXES]
    # Prefer files whose name suggests tensors/features/epochs/trials.
    tensor_candidates.sort(key=lambda p: (not any(k in p.name.lower() for k in ["tensor", "trial", "epoch", "feature", "meg"]), p.stat().st_size if p.exists() else 0))
    for p in tensor_candidates[: args.max_tensor_candidates]:
        loaded = try_load_tensor_file(p)
        if loaded is None:
            continue
        x, y, subjects = loaded
        if len(np.unique(subjects)) < 2 or len(np.unique(y)) < 2:
            continue
        x, y, subjects, meta = restrict_dataset(x, y, subjects, max_subjects=args.max_subjects, max_classes=args.max_classes, max_trials_per_subject_class=args.max_trials_per_subject_class, seed=args.seed)
        meta.update({"kind": "ready_tensor", "source_file": str(p)})
        return save_subset(args, x, y, subjects, meta)

    epoch_paths = find_mne_epoch_files(root)
    if epoch_paths:
        log(f"Found {len(epoch_paths)} MNE Epochs candidate files")
        loaded_epochs = load_mne_epochs(epoch_paths, max_subjects=args.max_subjects, tmin=args.tmin, tmax=args.tmax, resample_hz=args.resample_hz)
        if loaded_epochs is not None:
            x, y, subjects, source_meta = loaded_epochs
            x, y, subjects, meta = restrict_dataset(x, y, subjects, max_subjects=args.max_subjects, max_classes=args.max_classes, max_trials_per_subject_class=args.max_trials_per_subject_class, seed=args.seed)
            meta.update(source_meta)
            return save_subset(args, x, y, subjects, meta)

    raw_paths = find_mne_raw_files(root)
    if raw_paths:
        log(f"Found {len(raw_paths)} MNE Raw FIF candidate files")
        loaded_raw = load_mne_raw_events(
            raw_paths,
            max_subjects=args.max_subjects,
            max_classes=args.max_classes,
            max_trials_per_subject_class=args.max_trials_per_subject_class,
            event_code_max=args.event_code_max,
            tmin=args.tmin,
            tmax=args.tmax,
            resample_hz=args.resample_hz,
            seed=args.seed,
        )
        if loaded_raw is not None:
            x, y, subjects, source_meta = loaded_raw
            x, y, subjects, meta = restrict_dataset(
                x,
                y,
                subjects,
                max_subjects=args.max_subjects,
                max_classes=args.max_classes,
                max_trials_per_subject_class=args.max_trials_per_subject_class,
                seed=args.seed,
            )
            meta.update(source_meta)
            return save_subset(args, x, y, subjects, meta)

    raise RuntimeError(
        "No compatible cached dataset found. See cache_inventory.json for the files discovered under " + str(root)
    )


def run_baselines(dataset_path: Path, meta: dict, args) -> None:
    subjects = meta["selected_subjects"][: args.max_target_subjects]
    cmd = [
        sys.executable,
        "pytorch_meg_baselines.py",
        "--data",
        str(dataset_path),
        "--models",
        *args.models,
        "--target-subjects",
        *subjects,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--output-prefix",
        str(args.output_dir / "cached_public_meg_loso"),
        "--seed",
        str(args.seed),
    ]
    log("Running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--cache-root", type=Path, default=Path("/home/github-runner/.cache/datasets"))
    p.add_argument("--output-dir", type=Path, default=Path("results/cached_public_meg"))
    p.add_argument("--models", nargs="+", default=["eegnet", "lfcnn", "varcnn", "hgrn"])
    p.add_argument("--max-subjects", type=int, default=4)
    p.add_argument("--max-target-subjects", type=int, default=2)
    p.add_argument("--max-classes", type=int, default=16)
    p.add_argument("--max-trials-per-subject-class", type=int, default=30)
    p.add_argument("--max-tensor-candidates", type=int, default=100)
    p.add_argument("--event-code-max", type=int, default=1000)
    p.add_argument("--tmin", type=float, default=0.0)
    p.add_argument("--tmax", type=float, default=0.6)
    p.add_argument("--resample-hz", type=float, default=200.0)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.cache_root.exists():
        raise FileNotFoundError(f"Cache root does not exist: {args.cache_root}")
    dataset_path, meta = find_dataset(args)
    log("Selected cached dataset subset:")
    log(json.dumps(meta, indent=2)[:8000])
    run_baselines(dataset_path, meta, args)


if __name__ == "__main__":
    main()
