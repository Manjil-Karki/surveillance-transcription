#!/usr/bin/env python3
"""
evaluate_final.py
Generates all evaluation plots for the final report.
Loads trained M11 (ResNet18+Attention) and M09 (CNN AE+LSTM) checkpoints,
runs inference on the test set, and saves everything to outputs_final/.

Run from project root:
    python evaluate_final.py
"""

import os, sys, json, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, roc_auc_score, f1_score)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ─── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
os.chdir(ROOT)
OUT  = ROOT / 'outputs_final'
for d in ('plots', 'results'):
    (OUT / d).mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'GPU   : {p.name}  {p.total_memory/1e9:.1f} GB VRAM')

# ─── shared config ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

CLASSES      = ['Normal', 'RoadAccidents', 'Shoplifting', 'Arson', 'Burglary']
NUM_CLASSES  = len(CLASSES)
C2I          = {c: i for i, c in enumerate(CLASSES)}
I2C          = {i: c for c, i in C2I.items()}
CLASS_MAP    = {'NormalVideos': 'Normal', 'RoadAccidents': 'RoadAccidents',
                'Shoplifting': 'Shoplifting', 'Arson': 'Arson', 'Burglary': 'Burglary'}
COLORS = {
    'Normal':        '#4FC3F7',
    'RoadAccidents': '#29B6F6',
    'Shoplifting':   '#66BB6A',
    'Arson':         '#FF7043',
    'Burglary':      '#AB47BC',
}
CLIP_LEN   = 16
TEST_STRIDE = 4
DATASET_ROOT = ROOT / 'data' / 'dataset'

# ─── training history (from logged output) ────────────────────────────────────
M09_AE_TRAIN = [0.02100,0.01169,0.00997,0.00904,0.00840,0.00792,0.00756,
                0.00726,0.00702,0.00682,0.00664,0.00648,0.00634,0.00621,
                0.00610,0.00600,0.00591,0.00583,0.00575,0.00568,0.00561,
                0.00555,0.00549,0.00544,0.00539,0.00534,0.00530,0.00526,
                0.00522,0.00518]
M09_AE_VAL   = [0.01309,0.01049,0.00937,0.00854,0.00808,0.00764,0.00729,
                0.00701,0.00678,0.00661,0.00645,0.00631,0.00614,0.00601,
                0.00592,0.00585,0.00576,0.00566,0.00558,0.00555,0.00545,
                0.00551,0.00536,0.00534,0.00527,0.00522,0.00518,0.00513,
                0.00509,0.00505]
M09_JNT_F1  = [0.177,0.225,0.238,0.290,0.197,0.276,0.238,0.249,0.239,0.232,0.233]
M09_FTN_F1  = [0.238,0.255,0.245,0.241,0.265,0.268,0.259,0.276,0.250,0.245,0.231,0.238,0.230]
M11_P1_F1   = [0.321,0.324,0.298,0.285,0.281,0.333,0.243,0.282,0.343,0.282,
               0.290,0.344,0.359,0.318,0.319,0.321,0.400,0.351,0.286,0.302,
               0.324,0.326,0.351,0.325]
M11_P2A_F1  = [0.317,0.300,0.307,0.350,0.354,0.314,0.363,0.336]
M11_P2B_F1  = [0.294,0.314,0.299,0.296,0.291,0.298,0.316,0.337,0.312,
               0.297,0.299,0.308,0.309,0.310,0.291]

# ─── dataset utils ────────────────────────────────────────────────────────────
def _vid_id(stem):   return stem.rsplit('_', 1)[0]
def _frame_no(stem): return int(stem.rsplit('_', 1)[-1])

def discover_test():
    result = {}
    for folder, label in CLASS_MAP.items():
        fp = DATASET_ROOT / 'Test' / folder
        if not fp.exists(): continue
        groups = defaultdict(list)
        for f in fp.glob('*.png'):
            groups[_vid_id(f.stem)].append(f)
        for vid in groups:
            groups[vid].sort(key=lambda p: _frame_no(p.stem))
        result[label] = dict(groups)
    return result

def build_clips(vgroups, stride=TEST_STRIDE):
    clips, labels = [], []
    for lbl, vids in vgroups.items():
        idx = C2I[lbl]
        for frames in vids.values():
            for s in range(0, len(frames) - CLIP_LEN + 1, stride):
                clips.append(frames[s:s+CLIP_LEN])
                labels.append(idx)
    return clips, np.array(labels, np.int32)

# ─── M11 dataset ──────────────────────────────────────────────────────────────
IM_MEAN = [0.485, 0.456, 0.406]
IM_STD  = [0.229, 0.224, 0.225]
_tf_va  = T.Compose([T.Resize((112, 112)), T.ToTensor(), T.Normalize(IM_MEAN, IM_STD)])

class ClipDS11(Dataset):
    def __init__(self, clips, labels):
        self.clips = clips; self.labels = labels
    def __len__(self): return len(self.clips)
    def __getitem__(self, i):
        imgs = [Image.open(p).convert('RGB').resize((112,112), Image.BILINEAR)
                for p in self.clips[i]]
        x = torch.stack([_tf_va(im) for im in imgs])
        return x, torch.tensor(int(self.labels[i]), dtype=torch.long)

# ─── M11 model ────────────────────────────────────────────────────────────────
class ResNetEncoder11(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            *list(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())[:-1])
    def forward(self, x):
        B, T, C, H, W = x.shape
        return self.features(x.view(B*T,C,H,W)).view(B, T, 512)

class TemporalAttnHead11(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls  = nn.Parameter(torch.zeros(1,1,512))
        self.pos  = nn.Parameter(torch.zeros(1,CLIP_LEN+1,512))
        self.attn = nn.MultiheadAttention(512, num_heads=8, dropout=0.15, batch_first=True)
        self.norm1 = nn.LayerNorm(512); self.norm2 = nn.LayerNorm(512)
        self.ffn  = nn.Sequential(nn.Linear(512,1024),nn.GELU(),nn.Dropout(0.15),nn.Linear(1024,512))
        self.fc   = nn.Sequential(nn.Dropout(0.6),nn.Linear(512,128),nn.GELU(),
                                  nn.Dropout(0.6),nn.Linear(128,NUM_CLASSES))
    def forward(self, x):
        B = x.size(0)
        x = torch.cat([self.cls.expand(B,-1,-1), x], dim=1) + self.pos[:,:x.size(1)+1]
        a,_ = self.attn(x,x,x); x = self.norm1(x+a); x = self.norm2(x+self.ffn(x))
        return self.fc(x[:,0])

class VideoClassifier11(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNetEncoder11()
        self.head    = TemporalAttnHead11()
    def forward(self, x): return self.head(self.encoder(x))
    def encode(self, x):  return self.encoder(x).mean(dim=1)  # (B,512) mean pool for viz

# ─── M09 model (for confusion-matrix comparison) ──────────────────────────────
M09_FH = M09_FW = 64
M09_IN_CH = 6; M09_LATENT = 256; M09_LSTM = 128; M09_FC = 64; M09_DROP = 0.65
_M09_SP = M09_FH // 8; _M09_BTL = _M09_SP**2 * 128

def _load_frames_09(paths):
    out = []
    for p in paths:
        img = Image.open(p).convert('RGB').resize((M09_FW, M09_FH), Image.BILINEAR)
        out.append(np.array(img, np.float32)/255.0)
    return np.stack(out)

def _to_6ch(rgb):
    diff = np.diff(rgb, axis=0)
    diff = np.concatenate([np.zeros_like(rgb[:1]), diff], axis=0)
    return np.concatenate([rgb, diff], axis=-1)

class ClipDS09(Dataset):
    def __init__(self, clips, labels):
        self.clips = clips; self.labels = labels
    def __len__(self): return len(self.clips)
    def __getitem__(self, i):
        rgb = _load_frames_09(self.clips[i])
        x6  = _to_6ch(rgb)
        x   = torch.from_numpy(np.ascontiguousarray(x6.transpose(0,3,1,2))).float()
        return x, torch.tensor(int(self.labels[i]), dtype=torch.long)

class Encoder09(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(M09_IN_CH,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.MaxPool2d(2))
        self.fc = nn.Sequential(nn.Flatten(),nn.Linear(_M09_BTL,M09_LATENT),nn.ReLU())
    def forward(self, x):
        B,T,C,H,W = x.shape
        return self.fc(self.cnn(x.view(B*T,C,H,W))).view(B,T,M09_LATENT)

class Decoder09(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(M09_LATENT,_M09_BTL),nn.ReLU())
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,stride=2,padding=1),nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,stride=2,padding=1),nn.ReLU(),
            nn.ConvTranspose2d(32,3,4,stride=2,padding=1),nn.Sigmoid())
    def forward(self, z):
        B,T,_ = z.shape
        h = self.fc(z.view(B*T,M09_LATENT)).view(B*T,128,_M09_SP,_M09_SP)
        return self.deconv(h).view(B,T,3,M09_FH,M09_FW)

class LSTMHead09(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(M09_LATENT,M09_LSTM,batch_first=True)
        self.fc   = nn.Sequential(nn.Linear(M09_LSTM,M09_FC),nn.ReLU(),
                                  nn.Dropout(M09_DROP),nn.Linear(M09_FC,NUM_CLASSES))
    def forward(self, z):
        _,(h,_) = self.lstm(z); return self.fc(h[-1])

class VideoClassifier09(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder09(); self.decoder = Decoder09(); self.head = LSTMHead09()
    def forward(self, x):
        z = self.encoder(x); return self.decoder(z), self.head(z)

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Training curves
# ═══════════════════════════════════════════════════════════════════════════════
def plot_training_curves():
    print('Plotting training curves...')
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Training History: Model 09 vs Model 11', fontsize=14, fontweight='bold')

    # M09 Phase 1: AE loss
    ax = axes[0,0]
    ax.plot(M09_AE_TRAIN, color='#2196F3', lw=2, label='Train MSE')
    ax.plot(M09_AE_VAL,   color='#FF9800', lw=2, ls='--', label='Val MSE')
    ax.set_title('Model 09 — Phase 1: Autoencoder Pre-training', fontsize=10)
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    # M09 Phase 2+3: val F1
    ax = axes[0,1]
    ep2 = list(range(1, len(M09_JNT_F1)+1))
    ep3 = list(range(len(M09_JNT_F1)+1, len(M09_JNT_F1)+len(M09_FTN_F1)+1))
    ax.plot(ep2, M09_JNT_F1, color='#E91E63', lw=2, label='Phase 2: Joint CE+MSE', marker='o', ms=4)
    ax.plot(ep3, M09_FTN_F1, color='#9C27B0', lw=2, label='Phase 3: Pure CE', marker='s', ms=4)
    ax.axvline(len(M09_JNT_F1)+0.5, color='gray', ls=':', lw=1.2)
    ax.axhline(max(M09_JNT_F1), color='#E91E63', ls='--', alpha=0.5, lw=1)
    ax.text(len(M09_JNT_F1)+0.5, 0.06, 'Phase 3→', fontsize=8, color='gray')
    ax.set_title('Model 09 — Phases 2 & 3: Classification', fontsize=10)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Val Macro F1')
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 0.40)

    # M11 Phase 1: val F1
    ax = axes[1,0]
    ax.plot(list(range(1, len(M11_P1_F1)+1)), M11_P1_F1,
            color='#4CAF50', lw=2, marker='o', ms=4, label='Phase 1: Head only (frozen backbone)')
    best_ep = M11_P1_F1.index(max(M11_P1_F1)) + 1
    ax.axvline(best_ep, color='red', ls='--', lw=1.2, alpha=0.7)
    ax.scatter([best_ep], [max(M11_P1_F1)], color='red', zorder=5, s=80)
    ax.annotate(f'Best: {max(M11_P1_F1):.3f}\n(epoch {best_ep})',
                xy=(best_ep, max(M11_P1_F1)), xytext=(best_ep+2, max(M11_P1_F1)-0.04),
                fontsize=8, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1))
    ax.set_title('Model 11 — Phase 1: Backbone Frozen', fontsize=10)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Val Macro F1')
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0.15, 0.48)

    # M11 Phase 2a+2b: val F1
    ax = axes[1,1]
    ep_2a = list(range(1, len(M11_P2A_F1)+1))
    ep_2b = list(range(len(M11_P2A_F1)+1, len(M11_P2A_F1)+len(M11_P2B_F1)+1))
    ax.plot(ep_2a, M11_P2A_F1, color='#FF9800', lw=2, marker='o', ms=4,
            label='Phase 2a: Last block unlocked')
    ax.plot(ep_2b, M11_P2B_F1, color='#F44336', lw=2, marker='s', ms=4,
            label='Phase 2b: Layer3+4 unlocked')
    ax.axvline(len(M11_P2A_F1)+0.5, color='gray', ls=':', lw=1.2)
    ax.axhline(0.4002, color='#4CAF50', ls='--', alpha=0.5, lw=1.2,
               label=f'Phase 1 best (0.400)')
    ax.text(len(M11_P2A_F1)+0.5, 0.275, '2b→', fontsize=8, color='gray')
    ax.set_title('Model 11 — Phases 2a & 2b: Progressive Unfreezing', fontsize=10)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Val Macro F1')
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0.25, 0.43)

    plt.tight_layout()
    plt.savefig(OUT/'plots'/'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved training_curves.png')

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Ablation / Model Comparison
# ═══════════════════════════════════════════════════════════════════════════════
def plot_ablation():
    print('Plotting ablation comparison...')
    m09 = json.load(open(ROOT/'outputs_m09'/'results'/'result.json'))
    m11 = json.load(open(ROOT/'outputs_m11'/'results'/'result.json'))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Comparison: CNN Autoencoder+LSTM vs ResNet18+Attention', fontsize=13, fontweight='bold')

    # Overall metrics
    ax = axes[0]
    metrics = ['Macro F1', 'Accuracy', 'AUC-ROC']
    v09 = [m09['macro_f1'], m09['accuracy'], m09['auc_roc']]
    v11 = [m11['macro_f1'], m11['accuracy'], m11['auc_roc']]
    x   = np.arange(len(metrics)); w = 0.35
    b09 = ax.bar(x - w/2, v09, w, label='Model 09: CNN AE + LSTM', color='#78909C', alpha=0.85)
    b11 = ax.bar(x + w/2, v11, w, label='Model 11: ResNet18 + Attention (ours)', color='#42A5F5', alpha=0.9)
    for bar in b09:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, color='#455A64')
    for bar in b11:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1565C0')
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel('Score'); ax.set_ylim(0, 1.05)
    ax.set_title('Overall Metrics', fontsize=11)
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    # Per-class F1
    ax = axes[1]
    cls_labels = CLASSES
    v09c = [m09['per_class'][c] for c in cls_labels]
    v11c = [m11['per_class'][c] for c in cls_labels]
    x    = np.arange(len(cls_labels)); w = 0.35
    b09  = ax.bar(x - w/2, v09c, w, label='Model 09: CNN AE + LSTM',
                  color=[COLORS[c] for c in cls_labels], alpha=0.55)
    b11  = ax.bar(x + w/2, v11c, w, label='Model 11: ResNet18 + Attention',
                  color=[COLORS[c] for c in cls_labels], alpha=0.9)
    for bar, v in zip(b09, v09c):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{v:.2f}', ha='center', va='bottom', fontsize=8, color='#455A64')
    for bar, v in zip(b11, v11c):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{v:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(cls_labels, rotation=15, ha='right')
    ax.set_ylabel('F1 Score'); ax.set_ylim(0, 1.05)
    ax.set_title('Per-class F1', fontsize=11)
    patches = [mpatches.Patch(color='#78909C', alpha=0.7, label='Model 09 (faded)'),
               mpatches.Patch(color='#42A5F5', label='Model 11 (solid)')]
    ax.legend(handles=patches, fontsize=8); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT/'plots'/'ablation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved ablation_comparison.png')

# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE — run M11 and M09 on test set
# ═══════════════════════════════════════════════════════════════════════════════
def run_inference():
    print('\nLoading test clips...')
    vd_test = discover_test()
    te_clips, y_te = build_clips(vd_test)
    print(f'  Test clips: {len(te_clips):,}  |  classes: {np.bincount(y_te)}')

    # ── M11 ───────────────────────────────────────────────────────────────────
    print('\nLoading Model 11 (p2b_best.pth)...')
    m11 = VideoClassifier11().to(DEVICE)
    ckpt = ROOT / 'outputs_m11' / 'models' / 'p2b_best.pth'
    m11.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    m11.eval()

    ds11 = ClipDS11(te_clips, y_te)
    dl11 = DataLoader(ds11, batch_size=64, num_workers=4, pin_memory=True)

    preds11, probs11, embeds11, labs = [], [], [], []
    with torch.no_grad():
        for x, y in dl11:
            x = x.to(DEVICE, non_blocking=True)
            logits = m11(x)
            p = torch.softmax(logits, dim=1)
            emb = m11.encode(x)
            preds11.extend(logits.argmax(1).cpu().tolist())
            probs11.extend(p.cpu().numpy())
            embeds11.extend(emb.cpu().numpy())
            labs.extend(y.tolist())

    preds11  = np.array(preds11)
    probs11  = np.array(probs11)
    embeds11 = np.array(embeds11)
    labs     = np.array(labs)
    scores11 = 1.0 - probs11[:, C2I['Normal']]   # anomaly score

    # ── M09 ───────────────────────────────────────────────────────────────────
    print('Loading Model 09 (finetune_best.pth)...')
    m09 = VideoClassifier09().to(DEVICE)
    ckpt09 = ROOT / 'outputs_m09' / 'models' / 'finetune_best.pth'
    m09.load_state_dict(torch.load(ckpt09, map_location=DEVICE, weights_only=True))
    m09.eval()

    ds09 = ClipDS09(te_clips, y_te)
    dl09 = DataLoader(ds09, batch_size=64, num_workers=4, pin_memory=True)

    preds09, scores09 = [], []
    with torch.no_grad():
        for x, y in dl09:
            x = x.to(DEVICE, non_blocking=True)
            rec, logits = m09(x)
            mse = F.mse_loss(rec, x[:,:,:3], reduction='none').mean(dim=(1,2,3,4))
            preds09.extend(logits.argmax(1).cpu().tolist())
            scores09.extend(mse.cpu().tolist())
    preds09  = np.array(preds09)
    scores09 = np.array(scores09)

    return labs, preds11, probs11, scores11, embeds11, preds09, scores09

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Confusion matrices
# ═══════════════════════════════════════════════════════════════════════════════
def plot_confusion(labs, preds11, preds09):
    print('Plotting confusion matrices...')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle('Normalised Confusion Matrices — Test Set', fontsize=13, fontweight='bold')

    for ax, preds, title in zip(axes,
                                [preds09, preds11],
                                ['Model 09: CNN AE + LSTM (64×64)',
                                 'Model 11: ResNet18 + Attention (112×112)']):
        cm = confusion_matrix(labs, preds, normalize='true')
        im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASSES, rotation=20, ha='right', fontsize=9)
        ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASSES, fontsize=9)
        ax.set_xlabel('Predicted label'); ax.set_ylabel('True label')
        ax.set_title(title, fontsize=10)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                v = cm[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=9, color='white' if v > 0.55 else 'black')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(OUT/'plots'/'confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved confusion_matrices.png')

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — ROC curve + anomaly score distribution
# ═══════════════════════════════════════════════════════════════════════════════
def plot_anomaly(labs, scores11, scores09):
    print('Plotting anomaly detection...')
    y_bin = (labs != C2I['Normal']).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle('Anomaly Detection: Confidence-Based Score (1 − P(Normal))', fontsize=13, fontweight='bold')

    # ROC — both models
    ax = axes[0]
    fpr11, tpr11, _ = roc_curve(y_bin, scores11)
    fpr09, tpr09, _ = roc_curve(y_bin, scores09)
    auc11 = roc_auc_score(y_bin, scores11)
    auc09 = roc_auc_score(y_bin, scores09)
    ax.plot(fpr11, tpr11, color='#42A5F5', lw=2.5, label=f'M11 ResNet18+Attn  AUC={auc11:.3f}')
    ax.plot(fpr09, tpr09, color='#78909C', lw=1.8, ls='--', label=f'M09 CNN AE+LSTM   AUC={auc09:.3f}')
    ax.plot([0,1],[0,1],'k:',lw=1,label='Random (AUC=0.5)')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — Binary Anomaly Detection')
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xlim(0,1); ax.set_ylim(0,1.02)

    # Youden threshold for M11
    fpr11_, tpr11_, thr11_ = roc_curve(y_bin, scores11)
    j_idx  = np.argmax(tpr11_ - fpr11_)
    youden = thr11_[j_idx]

    # Score distribution — M11
    ax = axes[1]
    norm_scores = scores11[labs == C2I['Normal']]
    anom_scores = scores11[labs != C2I['Normal']]
    ax.hist(norm_scores, bins=60, alpha=0.6, color='#4FC3F7', density=True, label='Normal')
    ax.hist(anom_scores, bins=60, alpha=0.6, color='#EF5350', density=True, label='Anomaly')
    ax.axvline(youden, color='black', lw=1.8, ls='--', label=f'ROC-optimal τ={youden:.3f}')
    ax.set_xlabel('Anomaly Score (1 − P(Normal))')
    ax.set_ylabel('Density')
    ax.set_title('M11: Score Distribution')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Score distribution — M09
    ax = axes[2]
    fpr09_, tpr09_, thr09_ = roc_curve(y_bin, scores09)
    j09  = np.argmax(tpr09_ - fpr09_)
    y09  = thr09_[j09]
    n09  = scores09[labs == C2I['Normal']]
    a09  = scores09[labs != C2I['Normal']]
    ax.hist(n09, bins=60, alpha=0.6, color='#4FC3F7', density=True, label='Normal')
    ax.hist(a09, bins=60, alpha=0.6, color='#EF5350', density=True, label='Anomaly')
    ax.axvline(y09, color='black', lw=1.8, ls='--', label=f'ROC-optimal τ={y09:.4f}')
    ax.set_xlabel('Anomaly Score (MSE reconstruction error)')
    ax.set_ylabel('Density')
    ax.set_title('M09: Score Distribution (MSE)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT/'plots'/'anomaly_detection.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved anomaly_detection.png')

    # Threshold table
    thr_methods = [
        ('Mean + 2σ (Normal)', norm_scores.mean() + 2*norm_scores.std()),
        ('Mean + 3σ (Normal)', norm_scores.mean() + 3*norm_scores.std()),
        ('ROC-Optimal (Youden)',  youden),
    ]
    thr_results = []
    for name, tau in thr_methods:
        pred_bin = (scores11 >= tau).astype(int)
        tp = ((pred_bin==1)&(y_bin==1)).sum()
        fp = ((pred_bin==1)&(y_bin==0)).sum()
        fn = ((pred_bin==0)&(y_bin==1)).sum()
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        thr_results.append({'method': name, 'threshold': float(tau),
                            'precision': float(prec), 'recall': float(rec),
                            'f1': float(f1), 'auc': float(auc11)})
    with open(OUT/'results'/'threshold_comparison.json', 'w') as f:
        json.dump(thr_results, f, indent=2)
    return youden, thr_results

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Per-class F1 bar
# ═══════════════════════════════════════════════════════════════════════════════
def plot_per_class(labs, preds11):
    print('Plotting per-class metrics...')
    rep = classification_report(labs, preds11, target_names=CLASSES,
                                output_dict=True, zero_division=0)
    fig, ax = plt.subplots(figsize=(11, 5))
    x   = np.arange(NUM_CLASSES); w = 0.25
    P   = [rep[c]['precision'] for c in CLASSES]
    R   = [rep[c]['recall']    for c in CLASSES]
    F   = [rep[c]['f1-score']  for c in CLASSES]
    ax.bar(x - w, P, w, label='Precision', color='#42A5F5', alpha=0.85)
    ax.bar(x,     R, w, label='Recall',    color='#66BB6A', alpha=0.85)
    ax.bar(x + w, F, w, label='F1',        color='#FF7043', alpha=0.85)
    for xi, (p,r,f) in enumerate(zip(P,R,F)):
        ax.text(xi-w, p+0.01, f'{p:.2f}', ha='center', va='bottom', fontsize=8)
        ax.text(xi,   r+0.01, f'{r:.2f}', ha='center', va='bottom', fontsize=8)
        ax.text(xi+w, f+0.01, f'{f:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.axhline(rep['macro avg']['f1-score'], color='black', ls='--', lw=1.5,
               label=f'Macro F1={rep["macro avg"]["f1-score"]:.3f}')
    ax.set_xticks(x); ax.set_xticklabels(CLASSES, fontsize=10)
    ax.set_ylabel('Score'); ax.set_ylim(0, 1.05)
    ax.set_title('Model 11 — Per-class Precision, Recall, F1 (Test Set)', fontsize=12)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT/'plots'/'per_class_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved per_class_metrics.png')
    with open(OUT/'results'/'classification_report.json', 'w') as f:
        json.dump(rep, f, indent=2)

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 6 — PCA + t-SNE latent space
# ═══════════════════════════════════════════════════════════════════════════════
def plot_latent(embeds11, labs):
    print('Plotting latent space (PCA + t-SNE)...')
    N = min(2000, len(embeds11))
    idx = np.random.choice(len(embeds11), N, replace=False)
    E   = embeds11[idx]; L = labs[idx]
    clr = np.array([COLORS[I2C[l]] for l in L])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('ResNet18 Clip Embeddings — Latent Space', fontsize=13, fontweight='bold')

    # PCA
    pca = PCA(n_components=2, random_state=SEED)
    Ep  = pca.fit_transform(E)
    ax  = axes[0]
    for cls in CLASSES:
        mask = L == C2I[cls]
        ax.scatter(Ep[mask, 0], Ep[mask, 1], c=COLORS[cls], s=12, alpha=0.6, label=cls)
    ax.set_title(f'PCA (var explained: {pca.explained_variance_ratio_.sum()*100:.1f}%)', fontsize=11)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.legend(fontsize=9, markerscale=2); ax.grid(alpha=0.2)

    # t-SNE
    print('  Running t-SNE...')
    tsne = TSNE(n_components=2, perplexity=40, random_state=SEED, max_iter=1000)
    Et   = tsne.fit_transform(E)
    ax   = axes[1]
    for cls in CLASSES:
        mask = L == C2I[cls]
        ax.scatter(Et[mask, 0], Et[mask, 1], c=COLORS[cls], s=12, alpha=0.6, label=cls)
    ax.set_title('t-SNE (perplexity=40)', fontsize=11)
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=9, markerscale=2); ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUT/'plots'/'latent_space.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved latent_space.png')

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 7 — Timeline Gantt (synthetic sequence)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_timeline(labs, preds11, probs11, youden):
    print('Plotting timeline...')
    scores = 1.0 - probs11[:, C2I['Normal']]

    # Build synthetic sequence: 20 clips per segment
    SEG = 20
    sequence = []   # (class_label_idx, clips_from_test)
    for cls in ['Normal','RoadAccidents','Normal','Shoplifting','Normal','Arson','Normal','Burglary','Normal']:
        mask = np.where(labs == C2I[cls])[0]
        chosen = np.random.choice(mask, SEG, replace=len(mask) < SEG)
        sequence.extend(chosen)
    seq = np.array(sequence)

    # Sliding window majority vote (window=3)
    raw_preds = preds11[seq]
    raw_scores = scores[seq]
    smooth = []
    for i in range(len(raw_preds)):
        w = raw_preds[max(0,i-1):i+2]
        smooth.append(int(np.bincount(w, minlength=NUM_CLASSES).argmax()))
    smooth = np.array(smooth)

    # Merge consecutive segments
    segments = []
    i = 0
    while i < len(smooth):
        lbl = smooth[i]; j = i
        while j < len(smooth) and smooth[j] == lbl: j += 1
        seg_scores = raw_scores[i:j]
        mean_score = seg_scores.mean()
        is_anomaly = (lbl != C2I['Normal']) and (mean_score > youden)
        segments.append({'label': I2C[lbl], 'start': i, 'end': j-1,
                         'score': mean_score, 'anomaly': is_anomaly})
        i = j

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7),
                                    gridspec_kw={'height_ratios': [1.8, 1]})
    fig.suptitle('Synthetic Surveillance Session — Timeline & Anomaly Scores', fontsize=13, fontweight='bold')

    # Gantt chart
    for seg in segments:
        color = COLORS[seg['label']]
        ax1.barh(0, seg['end']-seg['start']+1, left=seg['start'],
                 height=0.5, color=color, alpha=0.85, edgecolor='white', lw=0.5)
        mid = (seg['start'] + seg['end']) / 2
        ax1.text(mid, 0, seg['label'][:6], ha='center', va='center',
                 fontsize=7.5, fontweight='bold', color='white')
        if seg['anomaly']:
            ax1.text(mid, 0.31, '⚠', ha='center', fontsize=10, color='red')

    handles = [mpatches.Patch(color=COLORS[c], label=c) for c in CLASSES]
    ax1.legend(handles=handles, loc='upper right', fontsize=8, ncol=5)
    ax1.set_xlim(0, len(smooth)); ax1.set_yticks([])
    ax1.set_ylabel('Event'); ax1.set_xlabel('')
    ax1.set_title('Predicted Event Timeline (3-clip majority-vote smoothing)', fontsize=10)
    ax1.grid(axis='x', alpha=0.3)

    # Score over time
    ax2.plot(raw_scores, color='#EF5350', lw=1.2, alpha=0.5, label='Score (raw)')
    # running mean
    k = 3
    pad = np.pad(raw_scores, (k//2, k//2), mode='edge')
    smooth_sc = np.convolve(pad, np.ones(k)/k, mode='valid')[:len(raw_scores)]
    ax2.plot(smooth_sc, color='#B71C1C', lw=2, label='Score (3-clip avg)')
    ax2.axhline(youden, color='black', ls='--', lw=1.5, label=f'Threshold τ={youden:.3f}')
    # shade anomaly regions
    for seg in segments:
        if seg['anomaly']:
            ax2.axvspan(seg['start'], seg['end']+1, alpha=0.12, color='red')
    ax2.set_xlim(0, len(smooth)); ax2.set_ylim(-0.02, 1.05)
    ax2.set_xlabel('Clip index'); ax2.set_ylabel('Anomaly Score')
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT/'plots'/'timeline.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved timeline.png')

    # Print timeline table
    print('\n  Timeline summary:')
    print(f'  {"Clips":<12} {"Class":<16} {"Score":>6}  {"Flag"}')
    for seg in segments:
        flag = '⚠ ANOMALY' if seg['anomaly'] else ''
        print(f'  {seg["start"]:>3}–{seg["end"]:<6}   {seg["label"]:<16} {seg["score"]:>6.3f}  {flag}')
    return segments

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 8 — Grad-CAM (M11)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_gradcam(m11_model, te_clips, labs):
    print('Plotting Grad-CAM...')
    target_layer = list(m11_model.encoder.features[6].children())[-1].conv2

    def get_cam(x_single, class_idx):
        # x_single: (T, C, H, W) — one clip
        # Hook fires on (B*T, C_feat, Hf, Wf) because encoder reshapes
        acts, grads = {}, {}
        def fwd_hook(m, inp, out):
            acts['f'] = out                 # (T, C_feat, Hf, Wf)
            out.register_hook(lambda g: grads.update({'g': g.detach()}))
        h = target_layer.register_forward_hook(fwd_hook)
        x_in = x_single.unsqueeze(0).to(DEVICE)   # (1,T,C,H,W)
        with torch.enable_grad():
            logits = m11_model(x_in)
            m11_model.zero_grad()
            logits[0, class_idx].backward()
        h.remove()
        # Average Grad-CAM over all T frames
        A = acts['f'].detach()             # (T, C_feat, Hf, Wf)
        G = grads['g']                     # (T, C_feat, Hf, Wf)
        weights = G.mean(dim=(0, 2, 3))    # (C_feat,) — mean over T and spatial
        A_mean  = A.mean(dim=0)            # (C_feat, Hf, Wf)
        cam = (weights[:, None, None] * A_mean).sum(0).clamp(min=0)   # (Hf, Wf)
        cam -= cam.min(); cam /= (cam.max() + 1e-8)
        cam_np = cam.cpu().numpy()         # 2D float array
        return cam_np

    # Pick 1 example per anomaly class
    fig, axes = plt.subplots(len(CLASSES)-1, 4, figsize=(14, 10))
    fig.suptitle('Grad-CAM Heatmaps — Model 11 (ResNet18 layer3)', fontsize=13, fontweight='bold')

    m11_model.eval()
    tf = T.Compose([T.Resize((112,112)), T.ToTensor(), T.Normalize(IM_MEAN, IM_STD)])
    tf_raw = T.Compose([T.Resize((112,112))])

    for row, cls in enumerate(CLASSES[1:]):  # skip Normal
        cls_idx = C2I[cls]
        candidates = [i for i, l in enumerate(labs) if l == cls_idx]
        if not candidates: continue
        clip_i = random.choice(candidates[:200])
        paths  = te_clips[clip_i]

        raw_frames = [Image.open(p).convert('RGB').resize((112,112), Image.BILINEAR) for p in paths]
        x = torch.stack([tf(im) for im in raw_frames])

        cam = get_cam(x, cls_idx)
        cam_img = Image.fromarray((cam*255).astype(np.uint8), mode='L')
        cam_up  = np.array(cam_img.resize((112,112), Image.BILINEAR))/255.0

        # Show 3 frames + CAM overlay on middle frame
        for col, frame_i in enumerate([0, 7, 15]):
            ax = axes[row, col]
            ax.imshow(raw_frames[frame_i])
            ax.set_title(f'{cls} — frame {frame_i}', fontsize=8)
            ax.axis('off')

        # Overlay
        ax = axes[row, 3]
        mid = raw_frames[7]
        overlay = np.array(mid).astype(float) / 255.0
        cmap = plt.cm.jet(cam_up)[..., :3]
        blended = 0.55 * overlay + 0.45 * cmap
        blended = np.clip(blended, 0, 1)
        ax.imshow(blended)
        ax.set_title(f'{cls} — Grad-CAM overlay', fontsize=8)
        ax.axis('off')

    col_labels = ['Frame 0', 'Frame 7 (mid)', 'Frame 15', 'Grad-CAM Overlay']
    for ax, label in zip(axes[0], col_labels):
        ax.set_title(label, fontsize=9, color='#1565C0', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUT/'plots'/'gradcam.png', dpi=130, bbox_inches='tight')
    plt.close()
    print('  Saved gradcam.png')

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 9 — EDA (updated class names)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_eda():
    print('Plotting EDA...')
    # Class clip counts from the actual split
    train_counts = {'Normal': 86335, 'RoadAccidents': 3939, 'Shoplifting': 4022,
                    'Arson': 2370, 'Burglary': 8558}
    val_counts   = {'Normal': 30998, 'RoadAccidents': 1504, 'Shoplifting': 2088,
                    'Arson': 3596, 'Burglary': 1024}
    test_counts  = {'Normal': 15730, 'RoadAccidents': 590, 'Shoplifting': 1836,
                    'Arson': 669, 'Burglary': 1871}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Dataset: UCF-Crime (5-Class Selection)', fontsize=13, fontweight='bold')

    # Stacked bar: train / val / test clip counts
    ax = axes[0]
    x  = np.arange(NUM_CLASSES); w = 0.6
    tr = [train_counts[c] for c in CLASSES]
    va = [val_counts[c]   for c in CLASSES]
    te = [test_counts[c]  for c in CLASSES]
    b1 = ax.bar(x, tr, w, label='Train', color=[COLORS[c] for c in CLASSES], alpha=0.9)
    b2 = ax.bar(x, va, w, bottom=tr, label='Val',
                color=[COLORS[c] for c in CLASSES], alpha=0.5)
    b3 = ax.bar(x, te, w, bottom=[a+b for a,b in zip(tr,va)], label='Test',
                color=[COLORS[c] for c in CLASSES], alpha=0.25)
    for i, (t,v,te_) in enumerate(zip(tr,va,te)):
        total = t+v+te_
        ax.text(i, total+500, f'{total:,}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=15, ha='right')
    ax.set_ylabel('Number of clips'); ax.set_title('Clip Counts per Split')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')
    ax.set_ylim(100, 200000)

    # Source video counts
    ax = axes[1]
    vid_counts = {'Normal': 800, 'RoadAccidents': 127, 'Shoplifting': 29, 'Arson': 41, 'Burglary': 87}
    vals = [vid_counts[c] for c in CLASSES]
    bars = ax.bar(CLASSES, vals, color=[COLORS[c] for c in CLASSES], alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of source videos')
    ax.set_title('Source Video Counts (diversity indicator)')
    ax.set_xticklabels(CLASSES, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT/'plots'/'eda_dataset.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved eda_dataset.png')

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    plot_training_curves()
    plot_ablation()
    plot_eda()

    labs, preds11, probs11, scores11, embeds11, preds09, scores09 = run_inference()

    plot_confusion(labs, preds11, preds09)
    youden, thr_results = plot_anomaly(labs, scores11, scores09)
    plot_per_class(labs, preds11)
    plot_latent(embeds11, labs)

    # Reload M11 for Grad-CAM (needs gradient-enabled model)
    print('Reloading M11 for Grad-CAM...')
    m11_gc = VideoClassifier11().to(DEVICE)
    m11_gc.load_state_dict(torch.load(
        ROOT/'outputs_m11'/'models'/'p2b_best.pth',
        map_location=DEVICE, weights_only=True))
    m11_gc.eval()
    vd_test = discover_test()
    te_clips_gc, labs_gc = build_clips(vd_test)
    plot_gradcam(m11_gc, te_clips_gc, labs_gc)
    plot_timeline(labs, preds11, probs11, youden)

    print(f'\nAll plots saved to {OUT}/plots/')
    print('Threshold comparison:')
    for t in thr_results:
        print(f"  {t['method']:<30} τ={t['threshold']:.3f}  P={t['precision']:.3f}"
              f"  R={t['recall']:.3f}  F1={t['f1']:.3f}  AUC={t['auc']:.3f}")
