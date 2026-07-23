"""
Evaluation: anomaly detection metrics, classification metrics, and all plots.
PyTorch version — models receive (B, T, C, H, W) tensors.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.config import (
    CLASSES, IDX_TO_CLASS, CLASS_TO_IDX, PLOTS_DIR, RESULTS_DIR, CLIP_LEN,
    FRAME_H, FRAME_W, CHANNELS
)

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _np_to_tensor(X: np.ndarray, device) -> torch.Tensor:
    """(N, T, H, W, C) numpy -> (N, T, C, H, W) tensor."""
    return torch.from_numpy(X.transpose(0, 1, 4, 2, 3)).float().to(device)


def _iter_batches(X, batch_size: int = 32, device=_DEVICE):
    """Yield (B, T, C, H, W) tensors from numpy array, packed .npy paths, or PNG path list."""
    if isinstance(X, np.ndarray):
        for i in range(0, len(X), batch_size):
            yield _np_to_tensor(X[i:i + batch_size], device)
    elif X and str(X[0]).endswith(".npy"):
        from src.dataset import PackedClipDataset
        from torch.utils.data import DataLoader
        loader = DataLoader(PackedClipDataset(X), batch_size=batch_size,
                            shuffle=False, num_workers=0)
        for clips in loader:
            yield clips.to(device)
    else:
        from src.dataset import ClipPathDataset
        from torch.utils.data import DataLoader
        loader = DataLoader(ClipPathDataset(X), batch_size=batch_size,
                            shuffle=False, num_workers=0)
        for clips in loader:
            yield clips.to(device)


def _tensor_to_np(t: torch.Tensor) -> np.ndarray:
    """(N, T, C, H, W) tensor -> (N, T, H, W, C) numpy."""
    return t.cpu().numpy().transpose(0, 1, 3, 4, 2)


# ── 1. Anomaly Scoring ────────────────────────────────────────────────────────

def compute_reconstruction_errors(model, X,
                                   batch_size: int = 32,
                                   device=_DEVICE) -> np.ndarray:
    """Per-clip mean MSE reconstruction error. X may be numpy array or path list."""
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in _iter_batches(X, batch_size, device):
            out   = model(batch)
            recon = out[0] if isinstance(out, tuple) else out
            mse   = ((recon - batch[:, :, :3, :, :]) ** 2).mean(dim=(1, 2, 3, 4))
            errors.extend(mse.cpu().tolist())
    return np.array(errors)


def compute_threshold(normal_errors: np.ndarray, n_std: float = 2.0) -> float:
    return float(normal_errors.mean() + n_std * normal_errors.std())


def find_optimal_threshold(errors: np.ndarray, binary_labels: np.ndarray) -> float:
    """Youden's J: maximise sensitivity + specificity - 1."""
    fpr, tpr, thresholds = roc_curve(binary_labels, errors)
    return float(thresholds[np.argmax(tpr - fpr)])


def threshold_comparison(encoder, decoder, X_val_normal: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          autoencoder=None, device=_DEVICE) -> tuple:
    from src.model import Autoencoder
    model = autoencoder if autoencoder is not None else Autoencoder(encoder, decoder)
    model = model.to(device)

    normal_errors = compute_reconstruction_errors(model, X_val_normal, device=device)
    test_errors   = compute_reconstruction_errors(model, X_test,       device=device)
    binary_gt     = (y_test != 0).astype(int)
    auc           = roc_auc_score(binary_gt, test_errors)

    results = {}
    for name, thr in [
        ("Mean + 2std", compute_threshold(normal_errors, 2.0)),
        ("Mean + 3std", compute_threshold(normal_errors, 3.0)),
        ("ROC-Optimal", find_optimal_threshold(test_errors, binary_gt)),
    ]:
        preds = (test_errors > thr).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            binary_gt, preds, average="binary", zero_division=0)
        results[name] = {"threshold": thr, "precision": p,
                         "recall": r, "f1": f, "auc": auc}

    _print_threshold_table(results)
    _save_json(results, "threshold_comparison.json")
    return results, normal_errors, test_errors, binary_gt


def _print_threshold_table(results: dict):
    print(f"\n{'Threshold':<15} {'Value':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<15} {r['threshold']:>8.5f} {r['precision']:>10.3f} "
              f"{r['recall']:>8.3f} {r['f1']:>8.3f} {r['auc']:>8.3f}")


# ── 2. Classification Metrics ─────────────────────────────────────────────────

def evaluate_classifier(joint_model, X_test, y_test: np.ndarray,
                         batch_size: int = 32, device=_DEVICE) -> tuple:
    """X_test may be numpy array or clip-path list."""
    joint_model.eval()
    preds = []
    with torch.no_grad():
        for batch in _iter_batches(X_test, batch_size, device):
            out    = joint_model(batch)
            logits = out[1] if isinstance(out, tuple) else out
            preds.extend(logits.argmax(dim=1).cpu().tolist())

    y_pred = np.array(preds)
    report = classification_report(
        y_test, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    print(classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0))
    _save_json(report, "classification_report.json")
    return y_pred, report


# ── 3. Latent Embeddings ──────────────────────────────────────────────────────

def get_embeddings(encoder, X, batch_size: int = 32,
                   device=_DEVICE) -> np.ndarray:
    """Mean-over-time encoder embeddings for PCA/t-SNE. X: numpy or path list."""
    encoder.eval()
    embs = []
    with torch.no_grad():
        for batch in _iter_batches(X, batch_size, device):
            e = encoder(batch)            # (B, T, LATENT_DIM)
            embs.append(e.mean(dim=1).cpu().numpy())
    return np.concatenate(embs, axis=0)


# ── 2b. Binary anomaly detection ──────────────────────────────────────────────

def evaluate_binary(joint_model, X_test, y_test: np.ndarray,
                    batch_size: int = 32, device=_DEVICE) -> tuple:
    """
    Binary anomaly detection: Normal (0) vs any Anomaly (1).
    Returns (results_dict, p_anomaly_scores, binary_gt).
    """
    normal_idx = CLASS_TO_IDX["Normal"]
    joint_model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in _iter_batches(X_test, batch_size, device):
            out    = joint_model(batch)
            logits = out[1] if isinstance(out, tuple) else out
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    all_probs  = np.concatenate(all_probs, axis=0)          # (N, C)
    p_anomaly  = 1.0 - all_probs[:, normal_idx]             # P(anomaly)
    binary_pred = (p_anomaly > 0.5).astype(int)
    binary_gt   = (y_test != normal_idx).astype(int)

    auc = roc_auc_score(binary_gt, p_anomaly)
    p, r, f, _ = precision_recall_fscore_support(
        binary_gt, binary_pred, average="binary", zero_division=0)
    acc = float((binary_pred == binary_gt).mean())

    result = {"accuracy": acc, "precision": p, "recall": r, "f1": f, "auc": auc}
    print("\nBinary Anomaly Detection  (Normal vs Anomaly)")
    print(f"  Accuracy : {acc:.3f}  Precision: {p:.3f}  Recall: {r:.3f}  F1: {f:.3f}  AUC: {auc:.3f}")
    _save_json(result, "binary_anomaly_report.json")
    return result, p_anomaly, binary_gt


# ── 2c. Fusion score ──────────────────────────────────────────────────────────

def compute_fusion_score(recon_errors: np.ndarray,
                          class_probs: np.ndarray,
                          alpha: float = 0.5) -> np.ndarray:
    """
    Blend normalised reconstruction error and classifier anomaly probability.
    alpha=1 -> recon only; alpha=0 -> classifier only; 0.5 -> equal blend.
    class_probs: (N, NUM_CLASSES) softmax output from joint model.
    Returns: (N,) fusion anomaly score in [0, 1].
    """
    norm_recon = (recon_errors - recon_errors.min()) / (recon_errors.ptp() + 1e-8)
    p_anomaly  = 1.0 - class_probs[:, CLASS_TO_IDX["Normal"]]
    return alpha * norm_recon + (1.0 - alpha) * p_anomaly


# ── 2d. Ablation comparison ───────────────────────────────────────────────────

def ablation_compare(models_dict: dict, X_test, y_test: np.ndarray,
                      batch_size: int = 32, device=_DEVICE) -> dict:
    """
    Evaluate multiple models on the same test set and print a summary table.
    models_dict: {name: model} — each model must output (recon, logits) or just logits.
    """
    results = {}
    for name, model in models_dict.items():
        print(f"\n{'='*45}\n{name}\n{'='*45}")
        y_pred, report = evaluate_classifier(model, X_test, y_test, batch_size, device)
        results[name] = {
            "accuracy":  report.get("accuracy", 0.0),
            "macro_f1":  report.get("macro avg", {}).get("f1-score", 0.0),
        }

    print("\n" + "="*55)
    print("ABLATION SUMMARY")
    print("="*55)
    print(f"{'Model':<35} {'Acc':>7} {'Macro F1':>10}")
    print("-"*55)
    for name, r in results.items():
        print(f"{name:<35} {r['accuracy']:>7.3f} {r['macro_f1']:>10.3f}")

    _save_json(results, "ablation_results.json")
    return results


# ── 4. Plots ──────────────────────────────────────────────────────────────────

def plot_roc_curve(test_errors, binary_gt,
                   save_path=PLOTS_DIR / "roc_curve.png"):
    fpr, tpr, _ = roc_curve(binary_gt, test_errors)
    auc = roc_auc_score(binary_gt, test_errors)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Anomaly Detection", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"[OK] ROC saved: {save_path}")


def plot_anomaly_scores(test_errors, y_test, threshold,
                        save_path=PLOTS_DIR / "anomaly_scores.png"):
    fig, ax = plt.subplots(figsize=(12, 3.5))
    x = np.arange(len(test_errors))
    nm = y_test == 0; an = ~nm
    ax.scatter(x[nm], test_errors[nm], s=6, alpha=0.5, color="steelblue", label="Normal")
    ax.scatter(x[an], test_errors[an], s=6, alpha=0.7, color="crimson",   label="Anomaly")
    ax.axhline(threshold, color="darkorange", ls="--", lw=1.5,
               label=f"Threshold = {threshold:.4f}")
    ax.set_xlabel("Clip Index"); ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Per-Clip Anomaly Scores", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"[OK] Anomaly scores saved: {save_path}")


def plot_confusion_matrix(y_true, y_pred,
                           save_path=PLOTS_DIR / "confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, data, fmt, title in [
        (axes[0], cm,      "d",   "Confusion Matrix (Counts)"),
        (axes[1], cm_norm, ".2f", "Confusion Matrix (Normalised)"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(title, fontweight="bold")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)
    plt.tight_layout(); fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"[OK] Confusion matrix saved: {save_path}")


def plot_reconstruction_samples(model, X: np.ndarray, n: int = 4,
                                  save_path=PLOTS_DIR / "reconstructions.png",
                                  device=_DEVICE):
    model.eval()
    idx    = np.random.choice(len(X), n, replace=False)
    sample = X[idx]                       # (n, T, H, W, C)
    batch  = _np_to_tensor(sample, device)
    with torch.no_grad():
        out   = model(batch)
        recon = out[0] if isinstance(out, tuple) else out
    recon_np = _tensor_to_np(recon)       # (n, T, H, W, C)

    fig, axes = plt.subplots(n, 2 * CLIP_LEN, figsize=(CLIP_LEN * 3, n * 2))
    for row in range(n):
        for col in range(CLIP_LEN):
            axes[row][col].imshow(sample[row, col, :, :, :3]); axes[row][col].axis("off")
            if row == 0: axes[row][col].set_title(f"In {col+1}", fontsize=7)
            axes[row][CLIP_LEN + col].imshow(np.clip(recon_np[row, col], 0, 1))
            axes[row][CLIP_LEN + col].axis("off")
            if row == 0: axes[row][CLIP_LEN + col].set_title(f"Recon {col+1}", fontsize=7)
    plt.suptitle("Input vs Reconstructed Frames", fontweight="bold")
    plt.tight_layout(); fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"[OK] Reconstructions saved: {save_path}")


def plot_mse_distribution(model, X_normal, X_anomaly,
                           save_path=PLOTS_DIR / "mse_distribution.png",
                           device=_DEVICE):
    norm_err = compute_reconstruction_errors(model, X_normal, device=device)
    anom_err = compute_reconstruction_errors(model, X_anomaly, device=device)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(norm_err, bins=60, alpha=0.6, color="steelblue", label="Normal",    density=True)
    ax.hist(anom_err, bins=60, alpha=0.6, color="crimson",   label="Anomalous", density=True)
    ax.set_xlabel("Reconstruction Error (MSE)"); ax.set_ylabel("Density")
    ax.set_title("MSE Distribution: Normal vs Anomalous", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"[OK] MSE distribution saved: {save_path}")


def plot_latent_space(embeddings, labels, method="pca",
                       save_path=PLOTS_DIR / "latent_space.png"):
    reducer = (TSNE(n_components=2, random_state=42, perplexity=30)
               if method == "tsne" else PCA(n_components=2, random_state=42))
    title   = "t-SNE of Latent Embeddings" if method == "tsne" else "PCA of Latent Embeddings"
    coords  = reducer.fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, cls in enumerate(CLASSES):
        mask = labels == i
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=12, alpha=0.6, color=_CLASS_COLORS.get(cls, "#888888"), label=cls)
    ax.set_title(title, fontweight="bold"); ax.legend(fontsize=9, markerscale=2)
    ax.grid(alpha=0.3); plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"[OK] Latent space saved: {save_path}")


_CLASS_COLORS = {
    "Normal":   "#4FC3F7",
    "Fighting": "#EF5350",
    "Robbery":  "#FF9800",
    "Arson":    "#FF7043",
    "Burglary": "#AB47BC",
}

def plot_binary_roc(p_anomaly: np.ndarray, binary_gt: np.ndarray,
                    save_path=PLOTS_DIR / "binary_roc.png"):
    fpr, tpr, _ = roc_curve(binary_gt, p_anomaly)
    auc = roc_auc_score(binary_gt, p_anomaly)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Binary Anomaly Detection ROC", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"[OK] Binary ROC saved: {save_path}")


def plot_per_class_metrics(report, save_path=PLOTS_DIR / "per_class_metrics.png"):
    metrics = {cls: report[cls] for cls in CLASSES if cls in report}
    x = np.arange(len(metrics)); w = 0.25
    prec = [metrics[c]["precision"] for c in metrics]
    rec  = [metrics[c]["recall"]    for c in metrics]
    f1   = [metrics[c]["f1-score"]  for c in metrics]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w, prec, w, label="Precision", color="#1E88E5")
    ax.bar(x,     rec,  w, label="Recall",    color="#43A047")
    ax.bar(x + w, f1,   w, label="F1-Score",  color="#FB8C00")
    ax.set_xticks(x); ax.set_xticklabels(list(metrics.keys()), rotation=20, ha="right")
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
    ax.set_title("Per-Class Classification Metrics", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"[OK] Per-class metrics saved: {save_path}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_json(data: dict, filename: str):
    out = RESULTS_DIR / filename
    with open(out, "w") as f:
        json.dump(data, f, indent=2, default=float)
    print(f"[OK] Saved: {out}")
