"""
Training loops — PyTorch.

Phase 1: Pre-train Encoder + Decoder on Normal clips (MSE loss).
Phase 3: Joint fine-tune Encoder + LSTM head (MSE + lambda * CrossEntropy).
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    PRETRAIN_EPOCHS, JOINT_EPOCHS, BATCH_SIZE, LEARNING_RATE,
    LAMBDA, SEED, MODELS_DIR, PLOTS_DIR
)
from src.dataset import make_loader

torch.manual_seed(SEED)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Phase 1: Pre-train Autoencoder ───────────────────────────────────────────

def pretrain_autoencoder(autoencoder, X_normal: np.ndarray,
                          val_split: float = 0.15,
                          device: torch.device = torch.device("cpu")) -> dict:
    n_val = int(len(X_normal) * val_split)
    X_tr, X_va = X_normal[n_val:], X_normal[:n_val]

    tr_loader = make_loader(X_tr, shuffle=True)
    va_loader = make_loader(X_va, shuffle=False)

    autoencoder = autoencoder.to(device)
    optimizer   = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion   = nn.MSELoss()

    best_val = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(PRETRAIN_EPOCHS):
        autoencoder.train()
        tr_losses = []
        for batch in tr_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = autoencoder(batch)
            loss  = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())

        autoencoder.eval()
        va_losses = []
        with torch.no_grad():
            for batch in va_loader:
                batch = batch.to(device)
                va_losses.append(criterion(autoencoder(batch), batch).item())

        tr_loss = float(np.mean(tr_losses))
        va_loss = float(np.mean(va_losses))
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        scheduler.step(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            patience_counter = 0
            torch.save(autoencoder.state_dict(), MODELS_DIR / "autoencoder_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= 7:
                print(f"[INFO] Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{PRETRAIN_EPOCHS}  "
                  f"loss={tr_loss:.5f}  val_loss={va_loss:.5f}")

    # Restore best and save components
    autoencoder.load_state_dict(torch.load(MODELS_DIR / "autoencoder_best.pth",
                                            map_location=device))
    torch.save(autoencoder.encoder.state_dict(), MODELS_DIR / "encoder_pretrained.pth")
    torch.save(autoencoder.decoder.state_dict(), MODELS_DIR / "decoder_pretrained.pth")
    _save_history(history, "pretrain_history.json")
    return history


# ── Phase 3: Joint Training ───────────────────────────────────────────────────

def joint_train(joint_model, X_train, y_train, X_val, y_val,
                lam: float = LAMBDA,
                device: torch.device = torch.device("cpu")) -> dict:

    tr_loader = make_loader(X_train, y_train, shuffle=True)
    va_loader = make_loader(X_val,   y_val,   shuffle=False)

    joint_model = joint_model.to(device)

    # Only optimise parameters with requires_grad=True (encoder + lstm_head)
    trainable   = [p for p in joint_model.parameters() if p.requires_grad]
    optimizer   = torch.optim.Adam(trainable, lr=LEARNING_RATE)
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    mse_loss    = nn.MSELoss()
    ce_loss     = nn.CrossEntropyLoss()

    best_val = float("inf")
    no_improve = 0
    history = {k: [] for k in [
        "train_loss", "train_recon", "train_cls", "train_acc",
        "val_loss",   "val_recon",   "val_cls",   "val_acc",
    ]}

    for epoch in range(JOINT_EPOCHS):
        joint_model.train()
        joint_model.decoder.eval()   # keep frozen decoder in eval mode

        tr = {k: [] for k in ["loss", "recon", "cls"]}
        tr_correct = tr_total = 0

        for clips, labels in tr_loader:
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad()
            recon, logits = joint_model(clips)
            r = mse_loss(recon, clips)
            c = ce_loss(logits, labels)
            loss = r + lam * c
            loss.backward()
            optimizer.step()
            tr["loss"].append(loss.item())
            tr["recon"].append(r.item())
            tr["cls"].append(c.item())
            tr_correct += (logits.argmax(1) == labels).sum().item()
            tr_total   += len(labels)

        joint_model.eval()
        va = {k: [] for k in ["loss", "recon", "cls"]}
        va_correct = va_total = 0
        with torch.no_grad():
            for clips, labels in va_loader:
                clips, labels = clips.to(device), labels.to(device)
                recon, logits = joint_model(clips)
                r = mse_loss(recon, clips)
                c = ce_loss(logits, labels)
                va["loss"].append((r + lam * c).item())
                va["recon"].append(r.item())
                va["cls"].append(c.item())
                va_correct += (logits.argmax(1) == labels).sum().item()
                va_total   += len(labels)

        tl = float(np.mean(tr["loss"]));  vl = float(np.mean(va["loss"]))
        ta = tr_correct / tr_total;       va_acc = va_correct / va_total
        history["train_loss"].append(tl);  history["val_loss"].append(vl)
        history["train_recon"].append(float(np.mean(tr["recon"])))
        history["val_recon"].append(float(np.mean(va["recon"])))
        history["train_cls"].append(float(np.mean(tr["cls"])))
        history["val_cls"].append(float(np.mean(va["cls"])))
        history["train_acc"].append(ta);   history["val_acc"].append(va_acc)

        scheduler.step(vl)

        if vl < best_val:
            best_val = vl
            no_improve = 0
            torch.save(joint_model.state_dict(), MODELS_DIR / "joint_model_best.pth")
        else:
            no_improve += 1
            if no_improve >= 7:
                print(f"[INFO] Early stopping at epoch {epoch + 1}")
                break

        print(f"Epoch {epoch+1:02d}/{JOINT_EPOCHS}  "
              f"loss={tl:.4f} recon={np.mean(tr['recon']):.4f} cls={np.mean(tr['cls']):.4f} "
              f"acc={ta:.3f} | val_loss={vl:.4f} val_acc={va_acc:.3f}")

    joint_model.load_state_dict(torch.load(MODELS_DIR / "joint_model_best.pth",
                                            map_location=device))
    _save_history(history, "joint_history.json")
    return history


def joint_train_lambda_sweep(X_train, y_train, X_val, y_val,
                              lambdas=(0.1, 0.5, 1.0),
                              device: torch.device = torch.device("cpu")) -> dict:
    from src.model import Encoder, Decoder, LSTMHead, JointModel, freeze_decoder
    results = {}
    for lam in lambdas:
        print(f"\n{'='*50}\nLambda = {lam}\n{'='*50}")
        enc  = Encoder(); dec = Decoder(); freeze_decoder(dec); lstm = LSTMHead()
        model = JointModel(enc, dec, lstm)
        enc.load_state_dict(torch.load(MODELS_DIR / "encoder_pretrained.pth", map_location=device))
        dec.load_state_dict(torch.load(MODELS_DIR / "decoder_pretrained.pth", map_location=device))
        hist = joint_train(model, X_train, y_train, X_val, y_val, lam=lam, device=device)
        results[lam] = {"val_loss": min(hist["val_loss"]), "val_acc": max(hist["val_acc"])}
        torch.save(model.state_dict(), MODELS_DIR / f"joint_model_lam{lam}.pth")
    print("\nLambda sweep results:")
    for lam, r in results.items():
        print(f"  λ={lam}  val_loss={r['val_loss']:.4f}  val_acc={r['val_acc']:.4f}")
    return results


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_loss(history: dict, title: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(history["train_loss"], label="Train", linewidth=1.5)
    ax.plot(history["val_loss"],   label="Val",   linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title(title, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Plot saved: {save_path}")


def plot_joint_loss(history: dict, save_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    pairs = [
        ("train_loss",  "val_loss",  "Total Joint Loss"),
        ("train_recon", "val_recon", "Reconstruction Loss (MSE)"),
        ("train_cls",   "val_cls",   "Classification Loss (CE)"),
    ]
    for ax, (tr_k, va_k, title) in zip(axes, pairs):
        ax.plot(history[tr_k], label="Train", linewidth=1.5)
        ax.plot(history[va_k], label="Val",   linewidth=1.5, linestyle="--")
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("Epoch"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle("Joint Training Loss Components", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Plot saved: {save_path}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_history(history: dict, filename: str):
    out = MODELS_DIR / filename
    with open(out, "w") as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.items()}, f, indent=2)
