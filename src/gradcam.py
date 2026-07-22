"""
Grad-CAM — PyTorch implementation using forward/backward hooks.
Targets the last Conv2d in the encoder CNN w.r.t. the LSTM classification score.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F

from src.config import (
    CLASSES, IDX_TO_CLASS, CLIP_LEN, FRAME_H, FRAME_W, PLOTS_DIR
)

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Grad-CAM class ────────────────────────────────────────────────────────────

class GradCAM:
    """
    Hook-based Grad-CAM on the last Conv2d of the encoder.

    Usage:
        gcam = GradCAM(encoder, lstm_head)
        heatmap, pred_cls, conf = gcam.compute(clip_np, class_idx, frame_idx)
        gcam.remove()
    """

    def __init__(self, encoder, lstm_head, device=_DEVICE):
        self.encoder   = encoder.to(device).eval()
        self.lstm_head = lstm_head.to(device).eval()
        self.device    = device
        self._acts     = None
        self._grads    = None

        # Hook the last Conv2d in encoder.cnn (index 6 in the Sequential)
        target = encoder.cnn[6]
        self._fwd_h = target.register_forward_hook(self._save_acts)
        self._bwd_h = target.register_full_backward_hook(self._save_grads)

    def _save_acts(self, module, inp, out):
        self._acts = out          # (B*T, 128, 8, 8)

    def _save_grads(self, module, grad_in, grad_out):
        self._grads = grad_out[0]  # (B*T, 128, 8, 8)

    def compute(self, clip_np: np.ndarray,
                class_idx: int = None,
                frame_idx: int = None) -> tuple:
        """
        Args:
            clip_np   : (T, H, W, C) float32 numpy array
            class_idx : target class (None -> predicted class)
            frame_idx : frame to visualise (None -> middle frame)
        Returns:
            heatmap   : (H, W) normalised Grad-CAM map
            pred_cls  : predicted class name
            conf      : confidence of pred_cls
        """
        if frame_idx is None:
            frame_idx = CLIP_LEN // 2

        # (1, T, C, H, W)
        clip_t = torch.from_numpy(
            clip_np.transpose(0, 3, 1, 2)
        ).float().unsqueeze(0).to(self.device)

        self.encoder.zero_grad()
        self.lstm_head.zero_grad()

        embeddings = self.encoder(clip_t)         # (1, T, LATENT_DIM)
        logits     = self.lstm_head(embeddings)   # (1, NUM_CLASSES)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        logits[0, class_idx].backward()

        # activations/grads: (1*T=T, 128, 8, 8) — pick frame_idx
        acts  = self._acts[frame_idx].detach()    # (128, 8, 8)
        grads = self._grads[frame_idx].detach()   # (128, 8, 8)

        weights = grads.mean(dim=(1, 2))           # (128,)
        cam = (weights[:, None, None] * acts).sum(dim=0)  # (8, 8)
        cam = F.relu(cam).cpu().numpy()

        if cam.max() > 0:
            cam /= cam.max()

        heatmap = np.array(
            Image.fromarray((cam * 255).astype(np.uint8))
                 .resize((FRAME_W, FRAME_H), Image.BILINEAR)
        ) / 255.0

        probs    = torch.softmax(logits, dim=1)[0]
        pred_cls = IDX_TO_CLASS[class_idx]
        conf     = float(probs[class_idx].item())

        return heatmap, pred_cls, conf

    def remove(self):
        self._fwd_h.remove()
        self._bwd_h.remove()


# ── Functional wrappers (match original API) ──────────────────────────────────

def compute_gradcam(encoder, lstm_head, clip: np.ndarray,
                    class_idx: int = None, frame_idx: int = None,
                    device=_DEVICE) -> tuple:
    gcam = GradCAM(encoder, lstm_head, device=device)
    result = gcam.compute(clip, class_idx, frame_idx)
    gcam.remove()
    return result


def overlay_heatmap(frame: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
    heatmap_rgb = cm.get_cmap("jet")(heatmap)[:, :, :3]
    return np.clip((1 - alpha) * frame + alpha * heatmap_rgb, 0, 1)


# ── Batch visualisation ───────────────────────────────────────────────────────

def visualize_gradcam_batch(encoder, lstm_head,
                             X: np.ndarray, y: np.ndarray,
                             n_per_class: int = 2,
                             save_path: Path = PLOTS_DIR / "gradcam_batch.png",
                             device=_DEVICE):
    from src.config import ANOMALY_CLASSES, CLASS_TO_IDX

    gcam   = GradCAM(encoder, lstm_head, device=device)
    n_rows = len(ANOMALY_CLASSES) * n_per_class
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, n_rows * 2.2))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    row = 0
    for cls in ANOMALY_CLASSES:
        cls_idx  = CLASS_TO_IDX[cls]
        pool     = np.where(y == cls_idx)[0]
        if len(pool) == 0:
            continue
        samples = np.random.choice(pool, min(n_per_class, len(pool)), replace=False)

        for s_idx in samples:
            clip     = X[s_idx]
            mid      = CLIP_LEN // 2
            frame    = clip[mid]
            heatmap, pred_cls, conf = gcam.compute(clip, class_idx=cls_idx, frame_idx=mid)
            overlay  = overlay_heatmap(frame, heatmap)

            axes[row][0].imshow(frame)
            axes[row][0].set_title(f"Input (True: {cls})", fontsize=8)
            axes[row][0].axis("off")
            axes[row][1].imshow(heatmap, cmap="jet")
            axes[row][1].set_title("Grad-CAM", fontsize=8)
            axes[row][1].axis("off")
            axes[row][2].imshow(overlay)
            axes[row][2].set_title(f"Overlay\nPred: {pred_cls} ({conf:.2f})", fontsize=8)
            axes[row][2].axis("off")
            row += 1

    gcam.remove()
    plt.suptitle("Grad-CAM: Spatial Attention per Anomaly Class",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Grad-CAM batch saved: {save_path}")
