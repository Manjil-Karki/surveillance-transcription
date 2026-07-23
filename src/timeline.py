"""
Timeline generation: sliding window inference, majority-vote smoothing,
segment merging, and Gantt-style visualisation.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import Counter

import torch

from src.config import (
    CLASSES, IDX_TO_CLASS, CLASS_TO_IDX, ANOMALY_CLASSES,
    CLIP_LEN, PLOTS_DIR
)

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_COLORS = {
    "Normal":   "#4FC3F7",   # sky blue
    "Fighting": "#EF5350",   # red
    "Robbery":  "#FF9800",   # orange
    "Arson":    "#FF7043",   # deep orange
    "Burglary": "#AB47BC",   # purple
}

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 1. Synthetic test sequence ────────────────────────────────────────────────

def build_test_sequence(X_test: np.ndarray, y_test: np.ndarray,
                         clips_per_segment: int = 20) -> tuple:
    """
    Interleave Normal and anomaly clips:
    Normal -> Class1 -> Normal -> Class2 -> ...
    """
    segments = []
    seq_clips, seq_labels = [], []

    normal_idx = np.where(y_test == CLASS_TO_IDX["Normal"])[0]
    np.random.shuffle(normal_idx)
    anomaly_order = [c for c in CLASSES if c != "Normal"]

    def _add_normal(n):
        pool = normal_idx[:n]
        for i in pool:
            seq_clips.append(X_test[i])
            seq_labels.append(CLASS_TO_IDX["Normal"])
        segments.append(("Normal", len(seq_clips) - n, len(seq_clips) - 1))

    _add_normal(clips_per_segment)
    for cls in anomaly_order:
        idx = CLASS_TO_IDX.get(cls)
        if idx is None:
            continue
        pool = np.where(y_test == idx)[0]
        if len(pool) == 0:
            continue
        start = len(seq_clips)
        for i in pool[:clips_per_segment]:
            seq_clips.append(X_test[i])
            seq_labels.append(idx)
        segments.append((cls, start, len(seq_clips) - 1))
        _add_normal(clips_per_segment)

    return (np.stack(seq_clips, axis=0),
            np.array(seq_labels, dtype=np.int32),
            segments)


# ── 2. Sliding-window inference ───────────────────────────────────────────────

def sliding_window_predict(joint_model, sequence_clips: np.ndarray,
                            batch_size: int = 32,
                            device=_DEVICE) -> tuple:
    """
    Run the joint model over every clip.
    Returns: pred_labels (N,), pred_probs (N, C), recon_errors (N,).
    pred_probs is needed for compute_fusion_score().
    """
    from src.evaluate import _np_to_tensor

    joint_model.eval()
    pred_labels, pred_probs, recon_errors = [], [], []

    with torch.no_grad():
        for i in range(0, len(sequence_clips), batch_size):
            batch_np  = sequence_clips[i: i + batch_size]
            batch     = _np_to_tensor(batch_np, device)
            recon, logits = joint_model(batch)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred_labels.extend(probs.argmax(axis=1).tolist())
            pred_probs.extend(probs.tolist())

            mse = ((recon - batch[:, :, :3, :, :]) ** 2).mean(dim=(1, 2, 3, 4)).cpu().numpy()
            recon_errors.extend(mse.tolist())

    return (np.array(pred_labels),
            np.array(pred_probs),
            np.array(recon_errors))


# ── 3. Smoothing ──────────────────────────────────────────────────────────────

def majority_vote_smooth(labels: np.ndarray, window: int = 3) -> np.ndarray:
    smoothed = labels.copy()
    half = window // 2
    for i in range(len(labels)):
        lo = max(0, i - half)
        hi = min(len(labels), i + half + 1)
        smoothed[i] = Counter(labels[lo:hi]).most_common(1)[0][0]
    return smoothed


# ── 4. Segment merging ────────────────────────────────────────────────────────

def merge_segments(labels: np.ndarray, recon_errors: np.ndarray,
                    threshold: float, fps: float = 1.0) -> list:
    clip_dur = CLIP_LEN
    segments, start = [], 0
    while start < len(labels):
        cls = labels[start]
        end = start
        while end + 1 < len(labels) and labels[end + 1] == cls:
            end += 1
        seg_errors = recon_errors[start: end + 1]
        mean_mse   = float(seg_errors.mean())
        is_anomaly = IDX_TO_CLASS[int(cls)] in ANOMALY_CLASSES
        segments.append({
            "class":      IDX_TO_CLASS[int(cls)],
            "start_clip": int(start),
            "end_clip":   int(end),
            "start_s":    int(start * clip_dur),
            "end_s":      int((end + 1) * clip_dur),
            "duration_s": int((end - start + 1) * clip_dur),
            "mean_mse":   round(mean_mse, 6),
            "is_anomaly": is_anomaly,
            "confirmed":  is_anomaly and (mean_mse > threshold),
        })
        start = end + 1
    return segments


# ── 5. Visualisation ──────────────────────────────────────────────────────────

def plot_timeline(segments: list, save_path: Path = PLOTS_DIR / "timeline.png"):
    fig, ax = plt.subplots(figsize=(14, 3.5))
    for seg in segments:
        color = CLASS_COLORS.get(seg["class"], "#AAAAAA")
        ax.barh(y=0, left=seg["start_s"], width=seg["duration_s"],
                height=0.5, color=color, edgecolor="black", linewidth=0.6)
        if seg["duration_s"] > 30:
            label = seg["class"] + (" [!]" if seg["confirmed"] else "")
            ax.text(seg["start_s"] + seg["duration_s"] / 2, 0, label,
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color="white" if seg["class"] != "Normal" else "black")
    patches = [mpatches.Patch(color=v, label=k, edgecolor="black", linewidth=0.5)
               for k, v in CLASS_COLORS.items()
               if any(s["class"] == k for s in segments)]
    ax.legend(handles=patches, loc="upper right", fontsize=8, ncol=3)
    ax.set_xlabel("Time (seconds)"); ax.set_yticks([])
    ax.set_title("Surveillance Event Timeline", fontsize=13, fontweight="bold")
    ax.set_xlim(0, segments[-1]["end_s"] if segments else 100)
    ax.grid(axis="x", alpha=0.3)
    for seg in segments:
        if seg["confirmed"]:
            ax.axvspan(seg["start_s"], seg["end_s"], alpha=0.15, color="red", zorder=0)
    plt.tight_layout(); fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"[OK] Timeline saved: {save_path}")


def plot_score_over_time(recon_errors, pred_labels, threshold,
                          save_path=PLOTS_DIR / "score_over_time.png"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    t = np.arange(len(recon_errors))
    ax1.plot(t, recon_errors, linewidth=0.8, color="#333333")
    ax1.axhline(threshold, color="red", ls="--", lw=1.2,
                label=f"Threshold ({threshold:.4f})")
    ax1.fill_between(t, recon_errors, threshold,
                     where=recon_errors > threshold, alpha=0.3, color="red")
    ax1.set_ylabel("Reconstruction MSE")
    ax1.set_title("Anomaly Score Over Time", fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    for i, cls in enumerate(CLASSES):
        mask = pred_labels == i
        ax2.scatter(t[mask], np.full(mask.sum(), i),
                    s=4, color=CLASS_COLORS.get(cls, "#888888"), label=cls, alpha=0.8)
    ax2.set_yticks(range(len(CLASSES))); ax2.set_yticklabels(CLASSES, fontsize=8)
    ax2.set_xlabel("Clip Index")
    ax2.set_title("Predicted Class Over Time", fontweight="bold")
    ax2.grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"[OK] Score-over-time saved: {save_path}")


def plot_fusion_over_time(fusion_scores: np.ndarray, pred_labels: np.ndarray,
                           threshold: float = 0.5,
                           save_path=PLOTS_DIR / "fusion_score_over_time.png"):
    """
    Plot combined reconstruction + classification anomaly score over clip index.
    fusion_scores: (N,) from evaluate.compute_fusion_score().
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    t = np.arange(len(fusion_scores))

    ax1.plot(t, fusion_scores, linewidth=0.8, color="#333333")
    ax1.axhline(threshold, color="red", ls="--", lw=1.2,
                label=f"Threshold ({threshold:.2f})")
    ax1.fill_between(t, fusion_scores, threshold,
                     where=fusion_scores > threshold, alpha=0.3, color="red")
    ax1.set_ylabel("Fusion Anomaly Score")
    ax1.set_title("Fusion Score Over Time (recon + classifier blend)", fontweight="bold")
    ax1.set_ylim(0, 1); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    for i, cls in enumerate(CLASSES):
        mask = pred_labels == i
        ax2.scatter(t[mask], np.full(mask.sum(), i),
                    s=4, color=CLASS_COLORS.get(cls, "#888888"), label=cls, alpha=0.8)
    ax2.set_yticks(range(len(CLASSES))); ax2.set_yticklabels(CLASSES, fontsize=8)
    ax2.set_xlabel("Clip Index")
    ax2.set_title("Predicted Class Over Time", fontweight="bold")
    ax2.grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"[OK] Fusion score saved: {save_path}")


def print_timeline(segments: list):
    print("\n" + "=" * 60)
    print("SURVEILLANCE EVENT TIMELINE")
    print("=" * 60)
    for seg in segments:
        flag = "  [ANOMALY CONFIRMED]" if seg["confirmed"] else ""
        s = _fmt(seg["start_s"]); e = _fmt(seg["end_s"])
        print(f"  {s} - {e}  |  {seg['class']:<12}  MSE={seg['mean_mse']:.5f}{flag}")
    print("=" * 60 + "\n")


def _fmt(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"
