"""
CNN-LSTM Multi-Task Autoencoder — PyTorch implementation.

Shared CNN encoder feeds:
  - Frozen decoder  (Task 1: Anomaly Detection via reconstruction MSE)
  - LSTM head       (Task 2: Event classification -> Timeline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    CLIP_LEN, FRAME_H, FRAME_W, CHANNELS,
    LATENT_DIM, LSTM_UNITS, FC_UNITS, DROPOUT, NUM_CLASSES
)

_SPATIAL = FRAME_H // 8        # 8  (after 3x MaxPool2d(2))
_BOTTLENECK = _SPATIAL ** 2 * 128  # 8192


# ── Encoder ───────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    TimeDistributed CNN: (B, T, C, H, W) -> (B, T, LATENT_DIM)

    Implemented via reshape trick: flatten time into batch, run CNN, reshape back.
    """
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 64->32
            nn.Conv2d(32, 64,  3, padding=1), nn.ReLU(), nn.MaxPool2d(2),        # 32->16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),        # 16->8
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(_BOTTLENECK, LATENT_DIM),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = self.cnn(x.view(B * T, C, H, W))
        x = self.fc(x)
        return x.view(B, T, LATENT_DIM)


# ── Decoder ───────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    TimeDistributed CNN decoder: (B, T, LATENT_DIM) -> (B, T, C, H, W)

    ConvTranspose2d with k=4, s=2, p=1 doubles spatial size:
      8 -> 16 -> 32 -> 64
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(LATENT_DIM, _BOTTLENECK),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),   # 8->16
            nn.ConvTranspose2d(64,  32, 4, stride=2, padding=1), nn.ReLU(),   # 16->32
            nn.ConvTranspose2d(32, CHANNELS, 4, stride=2, padding=1),
            nn.Sigmoid(),                                                        # 32->64
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x = self.fc(x.view(B * T, LATENT_DIM))
        x = x.view(B * T, 128, _SPATIAL, _SPATIAL)
        x = self.deconv(x)                          # (B*T, C, H, W)
        return x.view(B, T, CHANNELS, FRAME_H, FRAME_W)


# ── LSTM Classification Head ──────────────────────────────────────────────────

class LSTMHead(nn.Module):
    """LSTM: (B, T, LATENT_DIM) -> (B, NUM_CLASSES) logits"""

    def __init__(self):
        super().__init__()
        self.lstm    = nn.LSTM(LATENT_DIM, LSTM_UNITS, batch_first=True)
        self.fc      = nn.Sequential(
            nn.Linear(LSTM_UNITS, FC_UNITS),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FC_UNITS, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)   # h: (1, B, LSTM_UNITS)
        return self.fc(h.squeeze(0))


# ── Autoencoder (pre-training) ────────────────────────────────────────────────

class Autoencoder(nn.Module):
    """Encoder + Decoder for Phase 1 pre-training."""

    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ── Joint Model (fine-tuning) ─────────────────────────────────────────────────

class JointModel(nn.Module):
    """
    Shared encoder -> frozen decoder (Task 1) + LSTM head (Task 2).
    Returns (reconstruction, classification_logits).
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, lstm_head: LSTMHead):
        super().__init__()
        self.encoder   = encoder
        self.decoder   = decoder
        self.lstm_head = lstm_head

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings     = self.encoder(x)
        reconstruction = self.decoder(embeddings)
        classification = self.lstm_head(embeddings)
        return reconstruction, classification


# ── Single-frame CNN (Ablation Variant A) ────────────────────────────────────

class SingleFrameCNN(nn.Module):
    """Baseline: classify middle frame only, no temporal context."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64,  3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(_BOTTLENECK, 256), nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Utilities ─────────────────────────────────────────────────────────────────

def freeze_decoder(decoder: Decoder) -> None:
    for p in decoder.parameters():
        p.requires_grad = False
    n = sum(1 for p in decoder.parameters())
    print(f"[INFO] Decoder frozen ({n} parameter tensors).")


def unfreeze_decoder(decoder: Decoder) -> None:
    for p in decoder.parameters():
        p.requires_grad = True
