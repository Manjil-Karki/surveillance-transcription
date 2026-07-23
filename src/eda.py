"""EDA visualisations for the UCF-Crime pre-extracted PNG dataset."""

from pathlib import Path
from collections import defaultdict
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from src.config import (
    DATA_DIR, PLOTS_DIR, CLASSES, CLASS_MAP, CLASS_TO_IDX, ANOMALY_CLASSES,
    FRAME_H, FRAME_W, MAX_FRAMES, CLIP_LEN, TRAIN_STRIDE
)

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Consistent palette for the 5 selected classes
CLASS_COLORS = {
    "Normal":   "#4FC3F7",   # sky blue
    "Fighting": "#EF5350",   # red
    "Robbery":  "#FF9800",   # orange
    "Arson":    "#FF7043",   # deep orange
    "Burglary": "#AB47BC",   # purple
}
# Palette for all 14 classes (qualitative)
_PALETTE14 = [
    "#E53935","#8E24AA","#1E88E5","#00ACC1","#43A047",
    "#FB8C00","#F4511E","#6D4C41","#757575","#546E7A",
    "#D81B60","#3949AB","#039BE5","#00897B",
]

# ── helpers ──────────────────────────────────────────────────────────────────

def _count_frames(split: str) -> dict[str, int]:
    import os
    root = DATA_DIR / split
    counts = {}
    for d in sorted(root.iterdir()):
        if d.is_dir():
            counts[d.name] = sum(1 for e in os.scandir(d) if e.name.endswith(".png"))
    return counts


def _count_videos(split: str) -> dict[str, int]:
    import os
    root = DATA_DIR / split
    counts = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        vid_ids = set()
        for e in os.scandir(d):
            if not e.name.endswith(".png"):
                continue
            stem = e.name[:-4]  # strip .png
            m = re.match(r"^(.+?)_(\d+)$", stem)
            vid_ids.add(m.group(1) if m else stem)
        counts[d.name] = len(vid_ids)
    return counts


def _sample_frame(folder: Path) -> np.ndarray | None:
    pngs = sorted(folder.glob("*.png"))
    if not pngs:
        return None
    mid = pngs[len(pngs) // 2]
    return np.array(Image.open(mid).convert("RGB"))


# ── 1. Frame count overview (all 14 classes, both splits) ────────────────────

def plot_frame_counts(save: bool = True) -> tuple[dict, dict]:
    train_counts = _count_frames("Train")
    test_counts  = _count_frames("Test")
    all_classes  = sorted(train_counts.keys())

    selected = set(CLASS_MAP.keys())
    colors_train = [_PALETTE14[i % len(_PALETTE14)] if c in selected else "#CFD8DC"
                    for i, c in enumerate(all_classes)]
    colors_test  = [_PALETTE14[i % len(_PALETTE14)] if c in selected else "#ECEFF1"
                    for i, c in enumerate(all_classes)]

    x = np.arange(len(all_classes))
    w = 0.4

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w/2, [train_counts.get(c, 0) for c in all_classes],
           width=w, color=colors_train, label="Train")
    ax.bar(x + w/2, [test_counts.get(c,  0) for c in all_classes],
           width=w, color=colors_test,  label="Test", alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(all_classes, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Frame count")
    ax.set_title("Frame Counts per Class: Train vs Test", fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(handles=[
        mpatches.Patch(color=_PALETTE14[0], label="Train: selected"),
        mpatches.Patch(color="#CFD8DC",     label="Train: excluded"),
        mpatches.Patch(color="#ECEFF1",     label="Test"),
    ])
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(PLOTS_DIR / "eda_frame_counts.png", dpi=150, bbox_inches="tight")
    return train_counts, test_counts


# ── 2. Selected-class stats with capping ─────────────────────────────────────

def plot_selected_classes(train_counts: dict, save: bool = True):
    folder_names = list(CLASS_MAP.keys())
    raw    = [train_counts.get(f, 0) for f in folder_names]
    capped = [r if MAX_FRAMES is None else min(r, MAX_FRAMES) for r in raw]
    labels = [CLASS_MAP[f] for f in folder_names]
    x = np.arange(len(labels))
    w = 0.38

    cls_clrs = [CLASS_COLORS[l] for l in labels]

    cap_label = f"After cap ({MAX_FRAMES:,})" if MAX_FRAMES else "All frames (no cap)"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, raw,    width=w, color=cls_clrs, alpha=0.45, label="Raw", edgecolor="white")
    ax.bar(x + w/2, capped, width=w, color=cls_clrs, label=cap_label)
    if MAX_FRAMES:
        ax.axhline(MAX_FRAMES, color="red", ls="--", lw=1.2, label=f"Cap = {MAX_FRAMES:,}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Frame count")
    ax.set_title("Selected 5 Classes: Raw vs Full Frame Counts", fontweight="bold")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(PLOTS_DIR / "eda_selected_classes.png", dpi=150, bbox_inches="tight")


# ── 3. Sample frames grid ─────────────────────────────────────────────────────

def plot_sample_frames(n_per_class: int = 5, save: bool = True):
    folder_names = list(CLASS_MAP.keys())
    labels       = [CLASS_MAP[f] for f in folder_names]

    fig, axes = plt.subplots(len(labels), n_per_class,
                              figsize=(n_per_class * 1.6, len(labels) * 1.6))

    for row, (folder, label) in enumerate(zip(folder_names, labels)):
        pngs = sorted((DATA_DIR / "Train" / folder).glob("*.png"))
        step = max(1, len(pngs) // n_per_class)
        chosen = [pngs[i * step] for i in range(n_per_class) if i * step < len(pngs)]

        for col in range(n_per_class):
            ax = axes[row][col]
            ax.axis("off")
            if col < len(chosen):
                ax.imshow(np.array(Image.open(chosen[col]).convert("RGB")))
            if col == 0:
                ax.set_ylabel(label, fontsize=8, fontweight="bold",
                               rotation=0, labelpad=48, va="center")
            if row == 0:
                ax.set_title(f"Frame {col+1}", fontsize=7)

    plt.suptitle("Sample Frames per Selected Class", fontsize=11, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig(PLOTS_DIR / "eda_sample_frames.png", dpi=150, bbox_inches="tight")


# ── 4. Pixel brightness distribution per class ───────────────────────────────

def plot_brightness_distribution(n_samples: int = 200, save: bool = True):
    folder_names = list(CLASS_MAP.keys())
    labels       = [CLASS_MAP[f] for f in folder_names]

    fig, ax = plt.subplots(figsize=(9, 4))
    for folder, label in zip(folder_names, labels):
        pngs = sorted((DATA_DIR / "Train" / folder).glob("*.png"))
        step = max(1, len(pngs) // n_samples)
        sample = pngs[::step][:n_samples]
        brightnesses = [
            np.array(Image.open(p).convert("L")).mean() / 255.0
            for p in sample
        ]
        ax.hist(brightnesses, bins=30, alpha=0.55, label=label,
                color=CLASS_COLORS[label], edgecolor="none")

    ax.set_xlabel("Mean pixel brightness (normalised 0-1)")
    ax.set_ylabel("Frame count")
    ax.set_title("Brightness Distribution by Class", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(PLOTS_DIR / "eda_brightness.png", dpi=150, bbox_inches="tight")


# ── 5. Video count per class ──────────────────────────────────────────────────

def plot_video_counts(save: bool = True):
    vcounts = _count_videos("Train")
    folder_names = list(CLASS_MAP.keys())
    labels = [CLASS_MAP[f] for f in folder_names]
    vals   = [vcounts.get(f, 0) for f in folder_names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, vals, color=[CLASS_COLORS[l] for l in labels],
                  edgecolor="white", lw=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3, str(v), ha="center", fontsize=10)
    ax.set_ylabel("Unique video count")
    ax.set_title("Number of Unique Videos per Selected Class (Train)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(PLOTS_DIR / "eda_video_counts.png", dpi=150, bbox_inches="tight")
    return vcounts


# ── 6. Estimated clip counts after striding ───────────────────────────────────

def plot_clip_estimates(train_counts: dict, save: bool = True):
    folder_names = list(CLASS_MAP.keys())
    labels       = [CLASS_MAP[f] for f in folder_names]

    clips = []
    for f in folder_names:
        frames = train_counts.get(f, 0) if MAX_FRAMES is None else min(train_counts.get(f, 0), MAX_FRAMES)
        n_clips = max(0, (frames - CLIP_LEN) // TRAIN_STRIDE + 1)
        clips.append(n_clips)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, clips, color=[CLASS_COLORS[l] for l in labels],
                  edgecolor="white", lw=0.8)
    for bar, v in zip(bars, clips):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5, str(v), ha="center", fontsize=9)
    ax.set_ylabel(f"Estimated clips (stride={TRAIN_STRIDE}, len={CLIP_LEN})")
    ax.set_title("Estimated Training Clips per Class", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(PLOTS_DIR / "eda_clip_estimates.png", dpi=150, bbox_inches="tight")


# ── run all ───────────────────────────────────────────────────────────────────

def run_all_eda():
    print("Running EDA visualisations...")
    train_counts, test_counts = plot_frame_counts()
    plot_selected_classes(train_counts)
    plot_sample_frames()
    plot_brightness_distribution()
    plot_video_counts()
    plot_clip_estimates(train_counts)
    print(f"All EDA plots saved to {PLOTS_DIR}")
    return train_counts, test_counts
