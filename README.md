# Automatic Timeline Generation and Anomaly Flagging for Surveillance Video

**Module:** ST7088CEM – Artificial Neural Networks | **Student:** Manjil Karki

A ResNet18 + Temporal Attention classifier for surveillance video that simultaneously performs anomaly detection and descriptive event classification, producing a timestamped event timeline.

---

## Architecture

**Model 11 (final):** ResNet18 pretrained backbone → TimeDistributed reshape → CLS-token Transformer attention head → 5-class classifier. Anomaly score = `1 − P(Normal)`. Three-phase progressive unfreezing with FocalLoss(γ=2.5) + WeightedRandomSampler.

**Model 09 (baseline):** Custom 3-layer CNN encoder + frozen ConvTranspose decoder + LSTM(256) head. Joint MSE + CrossEntropy loss. Included as ablation comparison.

**Model 10 (intermediate, retired):** ResNet18 + LSTM at 64×64 resolution. Severe overfitting (train 98%, val 37%) due to spatial map collapse at low resolution. Prompted the switch to 112×112 and TemporalAttentionHead.

---

## Dataset

UCF-Crime Pre-extracted Image Dataset — 5 classes: Normal, RoadAccidents, Shoplifting, Arson, Burglary.  
Source: https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset

Place dataset at `data/ucf-crime-dataset/` before running.

---

## Results (test set, 20,696 clips)

| Model | Macro F1 | Accuracy | AUC-ROC |
|-------|:--------:|:--------:|:-------:|
| Model 09 (CNN AE + LSTM) | 0.335 | 0.734 | 0.591 |
| **Model 11 (ResNet18 + Attention)** | **0.382** | **0.762** | **0.853** |

---

## Notebooks

| Notebook | Description | Status |
|----------|-------------|--------|
| `notebooks/09_local_pipeline.ipynb` | CNN Autoencoder + LSTM full pipeline | Failure model (baseline) |
| `notebooks/10_resnet_pipeline.ipynb` | ResNet18 + LSTM at 64×64 | Failure model (retired) |
| `notebooks/11_resnet_pipeline.ipynb` | ResNet18 + TemporalAttentionHead | **Final model** |

---

## Scripts

```
python train_compare.py    # trains Model 09 and Model 11 on identical split
python evaluate_final.py   # loads checkpoints, generates 9 evaluation plots
```

---

## Requirements

```
pip install -r requirements.txt
```

Key dependencies: PyTorch 2.x, torchvision, scikit-learn, matplotlib, seaborn, Pillow.
