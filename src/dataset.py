"""
Data pipeline: clip building, loading, splitting, PyTorch Dataset.

Filename convention: Fighting002_x264_1020.png
  - video_id : Fighting002_x264
  - frame_no : 1020  (every 10th original frame)
"""

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import (
    CLASS_MAP, CLASS_TO_IDX, NUM_CLASSES, CLIP_LEN, TRAIN_STRIDE,
    TEST_STRIDE, MAX_FRAMES, FRAME_H, FRAME_W, CHANNELS,
    SEED, BATCH_SIZE, RESULTS_DIR,
    AUG_BRIGHTNESS, AUG_CONTRAST, AUG_NOISE_STD,
)

random.seed(SEED)
np.random.seed(SEED)


# ── 1. File discovery ─────────────────────────────────────────────────────────

def _get_frame_number(filename: str) -> int:
    stem = Path(filename).stem
    return int(stem.rsplit("_", 1)[-1])


def _get_video_id(filename: str) -> str:
    stem = Path(filename).stem
    return stem.rsplit("_", 1)[0]


def discover_videos(data_root: Path, split: str = "Train") -> dict:
    """
    Walk dataset folder and group PNG files by (label, video_id).
    Returns: {label: {video_id: [sorted Path list]}}
    """
    split_dir = data_root / split
    result = {}

    for folder_name, label in CLASS_MAP.items():
        folder = split_dir / folder_name
        if not folder.exists():
            print(f"[WARN] Folder not found: {folder}")
            continue

        video_groups = defaultdict(list)
        for f in folder.glob("*.png"):
            vid = _get_video_id(f.name)
            video_groups[vid].append(f)

        for vid in video_groups:
            video_groups[vid].sort(key=lambda p: _get_frame_number(p.name))

        result[label] = dict(video_groups)

    return result


# ── 2. Clip building ──────────────────────────────────────────────────────────

def build_clips(video_groups: dict, stride: int, max_frames: int = None) -> list:
    clips = []
    frame_count = 0
    video_ids = list(video_groups.keys())
    random.shuffle(video_ids)

    for vid in video_ids:
        frames = video_groups[vid]
        if max_frames and frame_count >= max_frames:
            break
        for start in range(0, len(frames) - CLIP_LEN + 1, stride):
            clips.append(frames[start: start + CLIP_LEN])
            frame_count += CLIP_LEN
            if max_frames and frame_count >= max_frames:
                break

    return clips


def build_all_clips(video_dict: dict, stride: int, cap: bool = True) -> tuple:
    X_paths, y = [], []
    for label, video_groups in video_dict.items():
        max_frames = MAX_FRAMES if cap else None
        clips = build_clips(video_groups, stride, max_frames)
        X_paths.extend(clips)
        y.extend([CLASS_TO_IDX[label]] * len(clips))
        print(f"  {label:<12} {len(clips):>5} clips")
    return X_paths, np.array(y, dtype=np.int32)


# ── 3. Frame loading ──────────────────────────────────────────────────────────

def _load_clip_rgb(clip_paths: list) -> np.ndarray:
    """Load frames as float32 RGB -> (T, H, W, 3) in [0, 1]. Uses cv2 for speed."""
    try:
        import cv2
        frames = []
        for p in clip_paths:
            img = cv2.imread(str(p))                                    # BGR uint8
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[:2] != (FRAME_H, FRAME_W):
                img = cv2.resize(img, (FRAME_W, FRAME_H), interpolation=cv2.INTER_LINEAR)
            frames.append(img.astype(np.float32) / 255.0)
    except (ImportError, Exception):
        frames = []
        for p in clip_paths:
            img = Image.open(p).convert("RGB").resize((FRAME_W, FRAME_H), Image.BILINEAR)
            frames.append(np.array(img, dtype=np.float32) / 255.0)
    return np.stack(frames, axis=0)   # (T, H, W, 3)


def _rgb_to_6ch(rgb: np.ndarray) -> np.ndarray:
    """(T, H, W, 3) RGB -> (T, H, W, 6) = RGB + frame-difference."""
    diffs = np.diff(rgb, axis=0)
    diffs = np.concatenate([np.zeros_like(rgb[:1]), diffs], axis=0)
    return np.concatenate([rgb, diffs], axis=-1)


def _augment_rgb(rgb: np.ndarray) -> np.ndarray:
    """Apply random spatial + photometric augmentations to (T, H, W, 3) array."""
    rgb = rgb.copy()

    if random.random() < 0.5:                                      # horizontal flip
        rgb = rgb[:, :, ::-1, :]

    if random.random() < 0.5:                                      # brightness jitter
        delta = random.uniform(-AUG_BRIGHTNESS, AUG_BRIGHTNESS)
        rgb = np.clip(rgb + delta, 0.0, 1.0)

    if random.random() < 0.5:                                      # contrast jitter
        factor = random.uniform(1.0 - AUG_CONTRAST, 1.0 + AUG_CONTRAST)
        mean = rgb.mean()
        rgb = np.clip(mean + factor * (rgb - mean), 0.0, 1.0)

    if random.random() < 0.3:                                      # Gaussian noise
        noise = np.random.normal(0.0, AUG_NOISE_STD, rgb.shape).astype(np.float32)
        rgb = np.clip(rgb + noise, 0.0, 1.0)

    return rgb


def load_clip(clip_paths: list) -> np.ndarray:
    """Load CLIP_LEN frames -> (T, H, W, 6): RGB + frame-difference channels."""
    return _rgb_to_6ch(_load_clip_rgb(clip_paths))



# ── 3b. Pre-packing (one .npy per clip for fast loading) ─────────────────────

def pack_clips_to_disk(X_paths: list, y: np.ndarray, out_dir: Path,
                       desc: str = "Packing") -> list:
    """
    Save each clip as a single uint8 RGB .npy file (16 PNGs -> 1 file).
    Returns list of .npy paths.
    ~40x faster to load than per-frame PNG reads.
    Disk cost: ~0.19 MB per clip (uint8 RGB only; diff computed at load time).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_paths = []
    for i, (clip_paths, label) in enumerate(
            tqdm(zip(X_paths, y), desc=desc, total=len(X_paths))):
        rgb_u8 = (_load_clip_rgb(clip_paths) * 255).astype(np.uint8)
        npy_path = out_dir / f"clip_{i:06d}_lbl{int(label)}.npy"
        np.save(npy_path, rgb_u8)
        npy_paths.append(npy_path)
    return npy_paths


class PackedClipDataset(Dataset):
    """
    Fast dataset backed by pre-packed per-clip .npy files.
    Each file stores (T, H, W, 3) uint8 RGB — diff channels computed at load time.
    Typical load time: ~0.2ms/clip vs ~8ms for PNG lazy loading.
    """

    def __init__(self, npy_paths: list, y: np.ndarray = None, training: bool = False):
        self.samples  = npy_paths
        self.y        = torch.from_numpy(y).long() if y is not None else None
        self.training = training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        rgb = np.load(self.samples[idx]).astype(np.float32) / 255.0  # (T, H, W, 3)
        if self.training:
            rgb = _augment_rgb(rgb)
        clip = _rgb_to_6ch(rgb)
        x    = torch.from_numpy(
            np.ascontiguousarray(clip.transpose(0, 3, 1, 2))
        ).float()
        if self.y is not None:
            return x, self.y[idx]
        return x


def make_packed_loader(npy_paths: list, y: np.ndarray = None,
                       shuffle: bool = True,
                       batch_size: int = BATCH_SIZE,
                       training: bool = False,
                       num_workers: int = 4) -> DataLoader:
    """DataLoader for pre-packed .npy clips. Fast and WSL2-safe."""
    ds = PackedClipDataset(npy_paths, y, training=training)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )


# ── 3c. Legacy bulk loader ────────────────────────────────────────────────────

def load_clips_to_array(X_paths: list, desc: str = "Loading",
                        num_workers: int = 4) -> np.ndarray:
    from concurrent.futures import ThreadPoolExecutor
    results = [None] * len(X_paths)
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(load_clip, p): i for i, p in enumerate(X_paths)}
        for f in tqdm(futures, desc=desc, total=len(X_paths)):
            results[futures[f]] = f.result()
    return np.stack(results, axis=0)   # (N, T, H, W, C)


# ── 4. Dataset splits ─────────────────────────────────────────────────────────

def make_video_splits(video_dict_train: dict, video_dict_test: dict,
                      val_ratio: float = 0.15):
    """
    Video-level split — zero inter-partition video overlap.

    Train + val: Train folder videos split by video ID (val_ratio held out).
    Test:        Test folder videos (entirely separate source videos).

    Returns:
        splits dict: {"train": (paths, y), "val": (paths, y), "test": (paths, y)}
        normal_train_vids: Normal video groups assigned to train (for X_normal.npy)
    """
    train_paths, y_train = [], []
    val_paths,   y_val   = [], []
    normal_train_vids    = {}

    for label, video_groups in video_dict_train.items():
        cls_idx  = CLASS_TO_IDX[label]
        vid_ids  = sorted(video_groups.keys())
        random.shuffle(vid_ids)

        n_val        = max(1, round(len(vid_ids) * val_ratio))
        val_vids     = {v: video_groups[v] for v in vid_ids[:n_val]}
        train_vids   = {v: video_groups[v] for v in vid_ids[n_val:]}

        if label == "Normal":
            normal_train_vids = train_vids

        tr = build_clips(train_vids, TRAIN_STRIDE, MAX_FRAMES)
        va = build_clips(val_vids,   TRAIN_STRIDE, MAX_FRAMES)

        train_paths.extend(tr); y_train.extend([cls_idx] * len(tr))
        val_paths.extend(va);   y_val.extend([cls_idx] * len(va))

        print(f"  {label:<12} train={len(tr):>4} clips ({len(vid_ids)-n_val} vids)  "
              f"val={len(va):>4} clips ({n_val} vids)")

    # Test set: UCF-Crime Test folder — videos never seen during training
    test_paths, y_test = [], []
    print("\nTest (separate Test folder):")
    for label, video_groups in video_dict_test.items():
        cls_idx = CLASS_TO_IDX[label]
        clips   = build_clips(video_groups, TEST_STRIDE, MAX_FRAMES)
        test_paths.extend(clips); y_test.extend([cls_idx] * len(clips))
        print(f"  {label:<12} test={len(clips):>4} clips")

    splits = {
        "train": (train_paths, np.array(y_train, dtype=np.int32)),
        "val":   (val_paths,   np.array(y_val,   dtype=np.int32)),
        "test":  (test_paths,  np.array(y_test,  dtype=np.int32)),
    }
    return splits, normal_train_vids


# ── 5. PyTorch Datasets ───────────────────────────────────────────────────────

class ClipDataset(Dataset):
    """
    Wraps numpy clip arrays for torch DataLoader.
    X stored as-is (N, T, H, W, C); transpose to (T, C, H, W) happens
    per item in __getitem__ so no full-array copy is made at init.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray = None, training: bool = False):
        self.X        = X   # (N, T, H, W, C) numpy — no copy
        self.y        = torch.from_numpy(y).long() if y is not None else None
        self.training = training

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        clip = self.X[idx]                      # (T, H, W, C)
        if self.training and random.random() < 0.5:
            clip = clip[:, :, ::-1, :]          # random horizontal flip
        x = torch.from_numpy(
            np.ascontiguousarray(clip.transpose(0, 3, 1, 2))
        ).float()
        if self.y is not None:
            return x, self.y[idx]
        return x


def make_loader(X: np.ndarray, y: np.ndarray = None,
                shuffle: bool = True, batch_size: int = BATCH_SIZE,
                training: bool = False) -> DataLoader:
    return DataLoader(ClipDataset(X, y, training=training), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, pin_memory=True)


class ClipPathDataset(Dataset):
    """
    Memory-efficient lazy-loading Dataset.
    Stores only clip frame paths; loads and processes each clip in __getitem__.
    RAM usage = one batch only, not the entire dataset.

    X_paths: list of clip_paths (each clip_paths is a list of Path objects).
    """

    def __init__(self, X_paths: list, y: np.ndarray = None, training: bool = False):
        self.samples  = X_paths
        self.y        = torch.from_numpy(y).long() if y is not None else None
        self.training = training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        rgb  = _load_clip_rgb(self.samples[idx])  # (T, H, W, 3)
        if self.training:
            rgb = _augment_rgb(rgb)
        clip = _rgb_to_6ch(rgb)                   # (T, H, W, 6)
        x    = torch.from_numpy(
            np.ascontiguousarray(clip.transpose(0, 3, 1, 2))
        ).float()
        if self.y is not None:
            return x, self.y[idx]
        return x


def _worker_init_fn(worker_id: int) -> None:
    seed = SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)


def make_path_loader(X_paths: list, y: np.ndarray = None,
                     shuffle: bool = True,
                     batch_size: int = BATCH_SIZE,
                     training: bool = False,
                     num_workers: int = 4) -> DataLoader:
    """DataLoader backed by lazy per-clip file loading. Low RAM footprint."""
    ds = ClipPathDataset(X_paths, y, training=training)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,   # avoid Jupyter/WSL2 worker deadlocks
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )


def compute_class_weights(y: np.ndarray) -> np.ndarray:
    """Balanced class weights for CrossEntropyLoss (inverse class frequency)."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.arange(NUM_CLASSES)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return weights.astype(np.float32)


# ── 6. Save / load preprocessed arrays ───────────────────────────────────────

def save_splits(splits: dict, normal_paths: list, out_dir: Path = RESULTS_DIR):
    import gc
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (X_paths, y) in splits.items():
        print(f"\nLoading {name} clips...")
        X = load_clips_to_array(X_paths, desc=name)
        np.save(out_dir / f"X_{name}.npy", X)
        np.save(out_dir / f"y_{name}.npy", y)
        print(f"  Saved X_{name}.npy {X.shape}")
        del X
        gc.collect()

    print("\nLoading Normal clips...")
    X_normal = load_clips_to_array(normal_paths, desc="Normal")
    np.save(out_dir / "X_normal.npy", X_normal)
    print(f"  Saved X_normal.npy {X_normal.shape}")
    del X_normal
    gc.collect()


def load_npy_split(split_name: str, out_dir: Path = RESULTS_DIR) -> tuple:
    X = np.load(out_dir / f"X_{split_name}.npy", mmap_mode="r")
    y = np.load(out_dir / f"y_{split_name}.npy")
    return X, y


def load_normal_npy(out_dir: Path = RESULTS_DIR) -> np.ndarray:
    return np.load(out_dir / "X_normal.npy", mmap_mode="r")
