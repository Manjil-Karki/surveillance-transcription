#!/usr/bin/env python3
"""
train_compare.py
Train Model-09 (CNN Autoencoder + LSTM) and Model-11 (ResNet18 + Attention)
back-to-back on the same data split, then print a side-by-side comparison.

Classes: Normal | RoadAccidents | Shoplifting | Arson | Burglary

Usage:
    python train_compare.py
"""

import os, sys, json, random, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score, classification_report

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  PROJECT ROOT & DEVICE
# ═══════════════════════════════════════════════════════════════════════════════

def _find_root():
    for p in [Path.cwd(), Path.cwd().parent]:
        if (p / 'data' / 'dataset').exists():
            return p
    raise FileNotFoundError(
        'Cannot find data/dataset/ — run from the project root '
        'or the notebooks/ subfolder')

ROOT = _find_root()
os.chdir(ROOT)
print(f'Project root : {ROOT}')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device       : {DEVICE}')
if torch.cuda.is_available():
    _p = torch.cuda.get_device_properties(0)
    print(f'GPU          : {_p.name}  {_p.total_memory/1e9:.1f} GB VRAM')

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  SHARED CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

CLASS_MAP = {
    'NormalVideos':  'Normal',
    'RoadAccidents': 'RoadAccidents',
    'Shoplifting':   'Shoplifting',
    'Arson':         'Arson',
    'Burglary':      'Burglary',
}
CLASSES         = list(CLASS_MAP.values())
NUM_CLASSES     = len(CLASSES)
CLASS_TO_IDX    = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS    = {i: c for c, i in CLASS_TO_IDX.items()}
ANOMALY_CLASSES = [c for c in CLASSES if c != 'Normal']

DATASET_ROOT = ROOT / 'data' / 'dataset'
CLIP_LEN     = 16
TRAIN_STRIDE = 8
TEST_STRIDE  = 4
VAL_SPLIT    = 0.25
N_PER_CLASS  = 5000   # clips drawn per class per epoch by WeightedRandomSampler

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  SHARED DATA DISCOVERY  (one pass, reused by both models)
# ═══════════════════════════════════════════════════════════════════════════════

def _vid_id(stem):    return stem.rsplit('_', 1)[0]
def _frame_no(stem):  return int(stem.rsplit('_', 1)[-1])

def discover_videos(split):
    split_dir = DATASET_ROOT / split
    result = {}
    for folder, label in CLASS_MAP.items():
        fp = split_dir / folder
        if not fp.exists():
            print(f'  [WARN] Not found: {fp}'); continue
        groups = defaultdict(list)
        for f in fp.glob('*.png'):
            groups[_vid_id(f.stem)].append(f)
        for vid in groups:
            groups[vid].sort(key=lambda p: _frame_no(p.stem))
        result[label] = dict(groups)
    return result

def build_clips(video_groups, stride):
    clips = []
    for frames in video_groups.values():
        for s in range(0, len(frames) - CLIP_LEN + 1, stride):
            clips.append(frames[s: s + CLIP_LEN])
    return clips

def make_splits(vd_train, vd_test):
    tr_clips, y_tr = [], []
    va_clips, y_va = [], []
    norm_train_vids = {}
    print('\nSplit summary:')
    for label, vgroups in vd_train.items():
        idx   = CLASS_TO_IDX[label]
        vids  = sorted(vgroups.keys()); random.shuffle(vids)
        n_val = max(1, round(len(vids) * VAL_SPLIT))
        val_v = {v: vgroups[v] for v in vids[:n_val]}
        tr_v  = {v: vgroups[v] for v in vids[n_val:]}
        if label == 'Normal': norm_train_vids = tr_v
        stride = TRAIN_STRIDE if label == 'Normal' else TEST_STRIDE
        tr = build_clips(tr_v, stride)
        va = build_clips(val_v, stride)
        tr_clips.extend(tr); y_tr.extend([idx] * len(tr))
        va_clips.extend(va); y_va.extend([idx] * len(va))
        print(f'  {label:<16} train={len(tr):>5}  val={len(va):>4}  ({len(vids)} vids)')
    te_clips, y_te = [], []
    for label, vgroups in vd_test.items():
        idx = CLASS_TO_IDX[label]
        te  = build_clips(vgroups, TEST_STRIDE)
        te_clips.extend(te); y_te.extend([idx] * len(te))
        print(f'  {label:<16} test={len(te):>5}')
    return (tr_clips, np.array(y_tr, np.int32),
            va_clips, np.array(y_va, np.int32),
            te_clips, np.array(y_te, np.int32),
            norm_train_vids)

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  MODEL 09 — CNN Autoencoder + LSTM
#     Input : (B, T, 6, 64, 64)  — RGB + frame-difference (motion channel)
#     Phases: 1) Pretrain AE on Normal only (MSE)
#             2) Joint MSE + CE, decoder frozen
#             3) Pure CE fine-tune, encoder unfrozen
# ═══════════════════════════════════════════════════════════════════════════════

OUT09 = ROOT / 'outputs_m09'
for _d in (OUT09/'models', OUT09/'plots', OUT09/'results'):
    _d.mkdir(parents=True, exist_ok=True)

# ── Model-09 hyper-parameters ──────────────────────────────────────────────
M09_FH = M09_FW = 64
M09_IN_CH   = 6        # RGB (3) + frame-diff (3)
M09_LATENT  = 256
M09_LSTM    = 128
M09_FC      = 64
M09_DROP    = 0.65
M09_BS      = 32
M09_EVAL_BS = 64
M09_WK      = 4
M09_PRE_EP  = 30       # autoencoder pretrain epochs
M09_JNT_EP  = 30       # joint MSE+CE epochs
M09_FTN_EP  = 15       # fine-tune pure-CE epochs
M09_LR      = 1e-4
M09_LAMBDA  = 0.5      # weight of MSE in joint loss
_M09_SP     = M09_FH // 8            # spatial size after 3× maxpool = 8
_M09_BTL    = _M09_SP ** 2 * 128     # bottleneck = 8192

# ── Model-09 data pipeline ─────────────────────────────────────────────────

def _load_frames_09(paths):
    out = []
    for p in paths:
        img = Image.open(p).convert('RGB')
        if img.size != (M09_FW, M09_FH):
            img = img.resize((M09_FW, M09_FH), Image.BILINEAR)
        out.append(np.array(img, np.float32) / 255.0)
    return np.stack(out)   # (T, H, W, 3)

def _to_6ch(rgb):
    diff = np.diff(rgb, axis=0)
    diff = np.concatenate([np.zeros_like(rgb[:1]), diff], axis=0)
    return np.concatenate([rgb, diff], axis=-1)   # (T, H, W, 6)

class ClipDS09(Dataset):
    def __init__(self, clips, y=None, training=False):
        self.clips = clips; self.y = y; self.training = training
    def __len__(self): return len(self.clips)
    def __getitem__(self, idx):
        rgb = _load_frames_09(self.clips[idx])
        if self.training:
            if random.random() < 0.5: rgb = rgb[::-1].copy()
            if random.random() < 0.5: rgb = rgb[:, :, ::-1, :].copy()
            if random.random() < 0.3:
                rgb = np.clip(rgb + np.random.normal(0, 0.02, rgb.shape).astype(np.float32), 0, 1)
        x6 = _to_6ch(rgb)
        x  = torch.from_numpy(np.ascontiguousarray(x6.transpose(0, 3, 1, 2))).float()
        if self.y is not None:
            return x, torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x

def _make_loader_09(clips, y=None, training=False, bs=M09_BS):
    ds = ClipDS09(clips, y, training)
    sampler = None
    if training and y is not None:
        counts  = np.maximum(np.bincount(y, minlength=NUM_CLASSES).astype(float), 1)
        weights = torch.from_numpy(1.0 / counts[y]).float()
        sampler = WeightedRandomSampler(weights, N_PER_CLASS * NUM_CLASSES, replacement=True)
    return DataLoader(ds, batch_size=bs, shuffle=(training and sampler is None),
                      sampler=sampler, num_workers=M09_WK, pin_memory=True,
                      persistent_workers=(M09_WK > 0))

# ── Model-09 architecture ──────────────────────────────────────────────────

class Encoder09(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(M09_IN_CH, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),         nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),        nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(_M09_BTL, M09_LATENT), nn.ReLU())
    def forward(self, x):
        B, T, C, H, W = x.shape
        return self.fc(self.cnn(x.view(B*T, C, H, W))).view(B, T, M09_LATENT)

class Decoder09(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(M09_LATENT, _M09_BTL), nn.ReLU())
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64,  32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32,   3, 4, stride=2, padding=1), nn.Sigmoid(),
        )
    def forward(self, z):
        B, T, _ = z.shape
        h = self.fc(z.view(B*T, M09_LATENT)).view(B*T, 128, _M09_SP, _M09_SP)
        return self.deconv(h).view(B, T, 3, M09_FH, M09_FW)

class LSTMHead09(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(M09_LATENT, M09_LSTM, batch_first=True)
        self.fc   = nn.Sequential(
            nn.Linear(M09_LSTM, M09_FC), nn.ReLU(),
            nn.Dropout(M09_DROP), nn.Linear(M09_FC, NUM_CLASSES),
        )
    def forward(self, z):
        _, (h, _) = self.lstm(z)
        return self.fc(h[-1])

class Autoencoder09(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder09(); self.decoder = Decoder09()
    def forward(self, x): return self.decoder(self.encoder(x))

class VideoClassifier09(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder09()
        self.decoder = Decoder09()
        self.head    = LSTMHead09()
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), self.head(z)

# ── Model-09 training helpers ──────────────────────────────────────────────

def _pretrain_ae09(ae, norm_clips, device):
    print('\n  [M09 Phase 1] Pretrain autoencoder on Normal clips')
    n_val  = max(1, int(len(norm_clips) * 0.15))
    tr_cl  = norm_clips[n_val:]; va_cl = norm_clips[:n_val]
    tr_l   = DataLoader(ClipDS09(tr_cl, training=True), batch_size=M09_BS,
                        shuffle=True, num_workers=M09_WK, pin_memory=True,
                        persistent_workers=M09_WK > 0)
    va_l   = DataLoader(ClipDS09(va_cl), batch_size=M09_EVAL_BS,
                        num_workers=M09_WK, pin_memory=True,
                        persistent_workers=M09_WK > 0)
    ae     = ae.to(device)
    opt    = torch.optim.Adam(ae.parameters(), lr=M09_LR)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    best   = float('inf'); no_imp = 0
    for ep in range(M09_PRE_EP):
        ae.train(); tr_l_vals = []
        for x in tr_l:
            x = x.to(device, non_blocking=True); opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                loss = F.mse_loss(ae(x), x[:, :, :3])
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tr_l_vals.append(loss.item())
        ae.eval(); va_l_vals = []
        with torch.no_grad():
            for x in va_l:
                x = x.to(device, non_blocking=True)
                va_l_vals.append(F.mse_loss(ae(x), x[:, :, :3]).item())
        tl = np.mean(tr_l_vals); vl = np.mean(va_l_vals)
        print(f'    AE {ep+1:02d}/{M09_PRE_EP}  mse={tl:.5f}  val_mse={vl:.5f}')
        if vl < best:
            best = vl; no_imp = 0
            torch.save(ae.state_dict(), OUT09 / 'models' / 'ae_best.pth')
        else:
            no_imp += 1
            if no_imp >= 5: print('    Early stop'); break
    ae.load_state_dict(torch.load(OUT09 / 'models' / 'ae_best.pth',
                                  map_location=device, weights_only=True))
    return ae

def _per_cls(va_preds, va_labs):
    return '  '.join(
        f'{IDX_TO_CLASS[i][0]}:'
        f'{sum(1 for p,l in zip(va_preds,va_labs) if p==l==i)/max(1,sum(1 for l in va_labs if l==i)):.2f}'
        for i in range(NUM_CLASSES))

def _train_joint09(model, tr_l, va_l, device, epochs, lam,
                   lr_enc, lr_head, tag, frozen_dec=True, patience=7):
    if frozen_dec:
        for p in model.decoder.parameters(): p.requires_grad = False
    opt = torch.optim.Adam([
        {'params': [p for p in model.encoder.parameters() if p.requires_grad], 'lr': lr_enc},
        {'params': list(model.head.parameters()), 'lr': lr_head},
    ], weight_decay=1e-4)
    scaler  = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    ce_fn   = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_f1 = 0.0; no_imp = 0
    for ep in range(epochs):
        model.train()
        for x, y in tr_l:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                rec, logits = model(x)
                ce_loss  = ce_fn(logits, y)
                mse_loss = F.mse_loss(rec, x[:, :, :3]) if lam > 0 else torch.tensor(0., device=device)
                loss     = lam * mse_loss + (1 - lam) * ce_loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt); scaler.update()
        model.eval(); va_preds, va_labs = [], []
        with torch.no_grad():
            for x, y in va_l:
                _, logits = model(x.to(device, non_blocking=True))
                va_preds.extend(logits.argmax(1).cpu().tolist())
                va_labs.extend(y.tolist())
        va_f1 = f1_score(va_labs, va_preds, average='macro', zero_division=0)
        print(f'    {tag} {ep+1:02d}/{epochs}  val_f1={va_f1:.3f}  [{_per_cls(va_preds, va_labs)}]')
        if va_f1 > best_f1:
            best_f1 = va_f1; no_imp = 0
            torch.save(model.state_dict(), OUT09 / 'models' / f'{tag}_best.pth')
        else:
            no_imp += 1
            if no_imp >= patience: print(f'    Early stop @ {ep+1}'); break
    model.load_state_dict(torch.load(OUT09 / 'models' / f'{tag}_best.pth',
                                     map_location=device, weights_only=True))
    print(f'    {tag} done. Best val F1: {best_f1:.4f}')

def run_model09(tr_clips, y_tr, va_clips, y_va, te_clips, y_te, norm_vids):
    print('\n' + '='*60 + '\nMODEL 09 — CNN Autoencoder + LSTM  (64×64, RGB+diff)\n' + '='*60)
    norm_clips = build_clips(norm_vids, TRAIN_STRIDE); random.shuffle(norm_clips)

    # Phase 1: pretrain autoencoder
    ae = _pretrain_ae09(Autoencoder09(), norm_clips, DEVICE)

    # Transfer weights into full classifier
    model = VideoClassifier09().to(DEVICE)
    model.encoder.load_state_dict(ae.encoder.state_dict())
    model.decoder.load_state_dict(ae.decoder.state_dict())

    tr_l = _make_loader_09(tr_clips, y_tr, training=True)
    va_l = _make_loader_09(va_clips, y_va, training=False, bs=M09_EVAL_BS)

    # Phase 2: joint MSE+CE
    print(f'\n  [M09 Phase 2] Joint loss (λ={M09_LAMBDA}), decoder frozen')
    _train_joint09(model, tr_l, va_l, DEVICE, M09_JNT_EP,
                   M09_LAMBDA, M09_LR, M09_LR, 'joint', frozen_dec=True)

    # Phase 3: pure CE, encoder unfrozen
    print('\n  [M09 Phase 3] Pure CE fine-tune, encoder unfrozen')
    for p in model.encoder.parameters(): p.requires_grad = True
    _train_joint09(model, tr_l, va_l, DEVICE, M09_FTN_EP,
                   0.0, 1e-5, 1e-4, 'finetune', frozen_dec=True, patience=5)

    # Evaluate on test set
    print('\n  [M09] Evaluating on test set...')
    model.eval(); all_preds, all_labs, all_scores = [], [], []
    te_l = _make_loader_09(te_clips, y_te, training=False, bs=M09_EVAL_BS)
    with torch.no_grad():
        for x, y in te_l:
            x   = x.to(DEVICE, non_blocking=True)
            rec, logits = model(x)
            mse = F.mse_loss(rec, x[:, :, :3], reduction='none').mean(dim=(1, 2, 3, 4))
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labs.extend(y.tolist())
            all_scores.extend(mse.cpu().tolist())

    rep   = classification_report(all_labs, all_preds, target_names=CLASSES,
                                  output_dict=True, zero_division=0)
    y_bin = (np.array(all_labs) != CLASS_TO_IDX['Normal']).astype(int)
    auc   = float(roc_auc_score(y_bin, all_scores)) if len(set(y_bin)) > 1 else 0.0
    res   = {'macro_f1': rep['macro avg']['f1-score'],
             'accuracy': rep['accuracy'], 'auc_roc': auc,
             'per_class': {c: rep[c]['f1-score'] for c in CLASSES}}
    with open(OUT09 / 'results' / 'result.json', 'w') as f:
        json.dump(res, f, indent=2, default=float)
    print(f'  Macro F1 : {res["macro_f1"]:.4f}')
    print(f'  Accuracy : {res["accuracy"]:.4f}')
    print(f'  AUC-ROC  : {res["auc_roc"]:.4f}')
    return res

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  MODEL 11 — ResNet18 + Temporal Attention
#     Input : (B, T, 3, 112, 112)
#     Phases: 1) Frozen backbone, attention head only
#             2a) Unfreeze last block of layer4
#             2b) Unfreeze layer3 + full layer4
# ═══════════════════════════════════════════════════════════════════════════════

OUT11 = ROOT / 'outputs_m11'
for _d in (OUT11/'models', OUT11/'plots', OUT11/'results'):
    _d.mkdir(parents=True, exist_ok=True)

# ── Model-11 hyper-parameters ──────────────────────────────────────────────
M11_FH = M11_FW = 112
M11_LATENT   = 512    # ResNet18 avgpool output
M11_FC       = 128
M11_DROP     = 0.6    # increased from 0.5 — Arson is smallest class, most oversampled
M11_BS       = 32
M11_EVAL_BS  = 64
M11_WK       = 12
M11_P1_EP    = 30
M11_P2A_EP   = 8
M11_P2B_EP   = 17
M11_LR_HEAD  = 1e-3
M11_LR_P2H   = 3e-4  # reduced head LR for phase 2 to stop oscillation
M11_LR_BACK  = 1e-5
IM_MEAN      = [0.485, 0.456, 0.406]
IM_STD       = [0.229, 0.224, 0.225]

_tf_tr11 = T.Compose([
    T.RandomResizedCrop(M11_FH, scale=(0.7, 1.0), ratio=(1.0, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    T.ToTensor(), T.Normalize(IM_MEAN, IM_STD),
])
_tf_va11 = T.Compose([
    T.Resize((M11_FH, M11_FW)),
    T.ToTensor(), T.Normalize(IM_MEAN, IM_STD),
])

def _motion_aug11(raw):
    if random.random() < 0.5: raw = raw[::-1]
    if random.random() < 0.3 and len(raw) > 2:
        i = random.randint(1, len(raw)-2); raw = list(raw); raw[i] = raw[i-1]
    if random.random() < 0.2 and len(raw) > 2:
        i = random.randint(0, len(raw)-2); raw = list(raw); raw[i+1] = raw[i]
    return raw

class ClipDS11(Dataset):
    def __init__(self, clips, y=None, training=False):
        self.clips = clips; self.y = y; self.training = training
    def __len__(self): return len(self.clips)
    def __getitem__(self, idx):
        raw = [Image.open(p).convert('RGB').resize((M11_FW, M11_FH), Image.BILINEAR)
               for p in self.clips[idx]]
        if self.training: raw = _motion_aug11(raw)
        seed = random.randint(0, 2**31)
        tf   = _tf_tr11 if self.training else _tf_va11
        tensors = []
        for img in raw:
            if self.training: random.seed(seed); torch.manual_seed(seed)
            tensors.append(tf(img))
        x = torch.stack(tensors)
        if self.training and random.random() < 0.2:
            x = x + torch.randn_like(x) * 0.05
        if self.y is not None:
            return x, torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x

def _make_loader_11(clips, y=None, training=False, bs=M11_BS):
    ds = ClipDS11(clips, y, training)
    sampler = None
    if training and y is not None:
        counts  = np.maximum(np.bincount(y, minlength=NUM_CLASSES).astype(float), 1)
        weights = torch.from_numpy(1.0 / counts[y]).float()
        sampler = WeightedRandomSampler(weights, N_PER_CLASS * NUM_CLASSES, replacement=True)
    return DataLoader(ds, batch_size=bs, shuffle=(training and sampler is None),
                      sampler=sampler, num_workers=M11_WK, pin_memory=True,
                      persistent_workers=(M11_WK > 0))

# ── Model-11 architecture ──────────────────────────────────────────────────

class ResNetEncoder11(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            *list(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())[:-1])
    def forward(self, x):
        B, T, C, H, W = x.shape
        return self.features(x.view(B*T, C, H, W)).view(B, T, M11_LATENT)
    def freeze_all(self):
        for p in self.parameters(): p.requires_grad = False
    def unfreeze_last_block(self):
        for n, p in self.named_parameters():
            if 'features.7.1' in n: p.requires_grad = True
    def unfreeze_layer3_4(self):
        for n, p in self.named_parameters():
            if 'features.6' in n or 'features.7' in n: p.requires_grad = True

class TemporalAttnHead11(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls  = nn.Parameter(torch.zeros(1, 1, M11_LATENT))
        self.pos  = nn.Parameter(torch.zeros(1, CLIP_LEN + 1, M11_LATENT))
        self.attn = nn.MultiheadAttention(M11_LATENT, num_heads=8,
                                           dropout=0.15, batch_first=True)
        self.norm1 = nn.LayerNorm(M11_LATENT)
        self.norm2 = nn.LayerNorm(M11_LATENT)
        self.ffn   = nn.Sequential(
            nn.Linear(M11_LATENT, M11_LATENT * 2), nn.GELU(),
            nn.Dropout(0.15), nn.Linear(M11_LATENT * 2, M11_LATENT),
        )
        self.fc    = nn.Sequential(
            nn.Dropout(M11_DROP), nn.Linear(M11_LATENT, M11_FC), nn.GELU(),
            nn.Dropout(M11_DROP), nn.Linear(M11_FC, NUM_CLASSES),
        )
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
    def forward(self, x):
        B  = x.size(0)
        x  = torch.cat([self.cls.expand(B, -1, -1), x], dim=1)   # (B, T+1, 512)
        x  = x + self.pos[:, :x.size(1)]
        a, _ = self.attn(x, x, x)
        x  = self.norm1(x + a)
        x  = self.norm2(x + self.ffn(x))
        return self.fc(x[:, 0])

class VideoClassifier11(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNetEncoder11()
        self.head    = TemporalAttnHead11()
    def forward(self, x): return self.head(self.encoder(x))

class FocalLoss11(nn.Module):
    def __init__(self, gamma=2.5):
        super().__init__(); self.gamma = gamma
    def forward(self, inp, tgt):
        ce   = F.cross_entropy(inp, tgt, reduction='none')
        loss = (1 - torch.exp(-ce)) ** self.gamma * ce
        return loss.mean()

# ── Model-11 training helper ───────────────────────────────────────────────

def _train_phase11(model, tr_l, va_l, opt, sched, epochs, tag, save_path, patience=7):
    crit   = FocalLoss11()
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))
    best_f1 = 0.0; no_imp = 0
    for ep in range(epochs):
        model.train()
        for x, y in tr_l:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt); scaler.update()
        sched.step()
        model.eval(); va_preds, va_labs = [], []
        with torch.no_grad():
            for x, y in va_l:
                va_preds.extend(model(x.to(DEVICE, non_blocking=True)).argmax(1).cpu().tolist())
                va_labs.extend(y.tolist())
        va_f1 = f1_score(va_labs, va_preds, average='macro', zero_division=0)
        print(f'    {tag} {ep+1:02d}/{epochs}  val_f1={va_f1:.3f}  [{_per_cls(va_preds, va_labs)}]')
        if va_f1 > best_f1:
            best_f1 = va_f1; no_imp = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_imp += 1
            if no_imp >= patience: print(f'    Early stop @ {ep+1}'); break
    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    print(f'    {tag} done. Best val F1: {best_f1:.4f}')

def run_model11(tr_clips, y_tr, va_clips, y_va, te_clips, y_te):
    print('\n' + '='*60 + '\nMODEL 11 — ResNet18 + Temporal Attention  (112×112)\n' + '='*60)
    tr_l = _make_loader_11(tr_clips, y_tr, training=True)
    va_l = _make_loader_11(va_clips, y_va, training=False, bs=M11_EVAL_BS)

    model = VideoClassifier11().to(DEVICE)
    model.encoder.freeze_all()
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Trainable params (head only): {n_tr:,}')

    # Phase 1
    print('\n  [M11 Phase 1] Backbone frozen — head only')
    opt1   = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                               lr=M11_LR_HEAD)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=M11_P1_EP)
    _train_phase11(model, tr_l, va_l, opt1, sched1, M11_P1_EP,
                   'P1', OUT11/'models'/'p1_best.pth')

    # Phase 2a
    print('\n  [M11 Phase 2a] Last block of layer4 unlocked')
    model.encoder.unfreeze_last_block()
    lb_p    = [p for n, p in model.encoder.named_parameters()
               if 'features.7.1' in n and p.requires_grad]
    opt2a   = torch.optim.Adam(
        [{'params': lb_p,                           'lr': M11_LR_BACK},
         {'params': list(model.head.parameters()),  'lr': M11_LR_P2H}])
    sched2a = torch.optim.lr_scheduler.CosineAnnealingLR(opt2a, T_max=M11_P2A_EP)
    _train_phase11(model, tr_l, va_l, opt2a, sched2a, M11_P2A_EP,
                   'P2a', OUT11/'models'/'p2a_best.pth', patience=5)

    # Phase 2b
    print('\n  [M11 Phase 2b] Layer3 + layer4 + head')
    model.encoder.unfreeze_layer3_4()
    l3_p = [p for n, p in model.encoder.named_parameters()
            if 'features.6' in n and p.requires_grad]
    l4_p = [p for n, p in model.encoder.named_parameters()
            if 'features.7' in n and p.requires_grad]
    opt2b   = torch.optim.Adam(
        [{'params': l3_p,                           'lr': M11_LR_BACK * 0.1},
         {'params': l4_p,                           'lr': M11_LR_BACK},
         {'params': list(model.head.parameters()),  'lr': M11_LR_P2H}])
    sched2b = torch.optim.lr_scheduler.CosineAnnealingLR(opt2b, T_max=M11_P2B_EP)
    _train_phase11(model, tr_l, va_l, opt2b, sched2b, M11_P2B_EP,
                   'P2b', OUT11/'models'/'p2b_best.pth')

    # Evaluate on test set
    print('\n  [M11] Evaluating on test set...')
    model.eval(); all_preds, all_labs, all_scores = [], [], []
    te_l = _make_loader_11(te_clips, y_te, training=False, bs=M11_EVAL_BS)
    with torch.no_grad():
        for x, y in te_l:
            logits = model(x.to(DEVICE, non_blocking=True))
            probs  = torch.softmax(logits, dim=1)
            scores = 1.0 - probs[:, CLASS_TO_IDX['Normal']]
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labs.extend(y.tolist())
            all_scores.extend(scores.cpu().tolist())

    rep   = classification_report(all_labs, all_preds, target_names=CLASSES,
                                  output_dict=True, zero_division=0)
    y_bin = (np.array(all_labs) != CLASS_TO_IDX['Normal']).astype(int)
    auc   = float(roc_auc_score(y_bin, all_scores)) if len(set(y_bin)) > 1 else 0.0
    res   = {'macro_f1': rep['macro avg']['f1-score'],
             'accuracy': rep['accuracy'], 'auc_roc': auc,
             'per_class': {c: rep[c]['f1-score'] for c in CLASSES}}
    with open(OUT11 / 'results' / 'result.json', 'w') as f:
        json.dump(res, f, indent=2, default=float)
    print(f'  Macro F1 : {res["macro_f1"]:.4f}')
    print(f'  Accuracy : {res["accuracy"]:.4f}')
    print(f'  AUC-ROC  : {res["auc_roc"]:.4f}')
    return res

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def print_comparison(r09, r11):
    w = lambda a, b: 'M09 ✓' if a > b else ('M11 ✓' if b > a else 'tie')
    print('\n' + '='*60)
    print('FINAL COMPARISON — Model 09 vs Model 11')
    print('Classes: ' + ' | '.join(CLASSES))
    print('='*60)
    print(f'  {"Metric":<20} {"Model-09":>10} {"Model-11":>10}  {"Winner":>8}')
    print('  ' + '-'*52)
    for label, k09, k11 in [
        ('Macro F1',  r09['macro_f1'], r11['macro_f1']),
        ('Accuracy',  r09['accuracy'], r11['accuracy']),
        ('AUC-ROC',   r09['auc_roc'],  r11['auc_roc']),
    ]:
        print(f'  {label:<20} {k09:>10.4f} {k11:>10.4f}  {w(k09,k11):>8}')
    print('\n  Per-class F1:')
    print(f'  {"Class":<16} {"M09":>8} {"M11":>8}  {"Winner":>8}')
    print('  ' + '-'*44)
    for c in CLASSES:
        v09 = r09['per_class'].get(c, 0.0)
        v11 = r11['per_class'].get(c, 0.0)
        print(f'  {c:<16} {v09:>8.4f} {v11:>8.4f}  {w(v09,v11):>8}')
    print('='*60)
    out = ROOT / 'comparison_result.json'
    with open(out, 'w') as f:
        json.dump({'model09': r09, 'model11': r11}, f, indent=2, default=float)
    print(f'Saved → {out}')

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()

    print('\nDiscovering dataset...')
    vd_train = discover_videos('Train')
    vd_test  = discover_videos('Test')
    (tr_clips, y_tr, va_clips, y_va,
     te_clips, y_te, norm_vids) = make_splits(vd_train, vd_test)

    r09 = run_model09(tr_clips, y_tr, va_clips, y_va, te_clips, y_te, norm_vids)
    r11 = run_model11(tr_clips, y_tr, va_clips, y_va, te_clips, y_te)

    print_comparison(r09, r11)
    elapsed = time.time() - t0
    print(f'\nTotal wall time: {elapsed/3600:.1f} h  ({elapsed/60:.0f} min)')
