# EchoRisk — MICCAI 2026 Challenge

Cardio-oncology risk prediction from echocardiography video (DICOM).  
Three tasks using a shared R(2+1)D-18 backbone with dual-view (A4C + A2C) fusion.

---

## Tasks

| # | Name | Output | Metric | Train / Val |
|---|------|--------|--------|-------------|
| 1 | LVEF Regression | Ejection fraction (%) | MAE ↓ (target ≤ 5%) | 630 / 156 exams |
| 2 | LV Dysfunction Classification | Binary logit | AUC-ROC ↑ | 630 / 156 exams |
| 3 | Cardiotoxicity Prediction (T1 only) | Binary logit | AUC-ROC ↑ | 157 / 39 patients |

---

## Project Structure

```
echo_risk/
├── src/
│   ├── dataset.py        # DICOM loader + EchoDataset (all tasks)
│   ├── model.py          # EchoNet — R(2+1)D-18 + dual-view fusion head
│   ├── train_task1.py    # LVEF regression training
│   ├── train_task2.py    # LV dysfunction classification training
│   └── train_task3.py    # Cardiotoxicity prediction (transfer learning)
├── data/
│   ├── task1/Task1/public/{dicom,labels}/
│   ├── task2/Task2/public/{dicom,labels}/
│   └── task3/Task3/public/{dicom,labels}/
├── checkpoints/
│   ├── task1/best.pth
│   ├── task2/best.pth
│   └── task3/best.pth
└── requirements.txt
```

---

## Architecture

**EchoNet** — shared backbone, task-specific head:

- **Backbone**: `R(2+1)D-18` pretrained on Kinetics-400, FC layer removed → 512-d feature per view
- **Dual-view fusion**: A4C and A2C processed independently; A2C features gated to zero when the view is absent (`has_a2c = 0`)
- **Head**: `Flatten → Linear(1024→256) → ReLU → Dropout(0.3) → Linear(256→1)`
- **Output**: raw scalar (LVEF %) for regression; raw logit for classification tasks

```
A4C video (B,1,T,H,W) ──► backbone ──► f4 (B,512) ──┐
                                                       cat ──► MLP head ──► output (B,)
A2C video (B,1,T,H,W) ──► backbone ──► f2 (B,512) ──┘
                                         × has_a2c
```

---

## Setup

```bash
conda create -n challenge python=3.11 -y
conda activate challenge

# PyTorch with CUDA 12.4 (requires driver ≥ 525)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Remaining dependencies
pip install -r requirements.txt
```

> The `requirements.txt` deliberately excludes torch/torchvision so the CUDA build above
> is not overwritten by a CPU-only wheel from PyPI.

---

## Training

Run tasks in order — Task 3 loads a backbone pretrained by Task 1 or 2.

### Task 1 — LVEF Regression

```bash
cd src
python train_task1.py
# or override defaults:
python train_task1.py --epochs 80 --batch_size 8 --lr 5e-5
```

Key design choices:
- **Huber loss** (`delta=5.0`) — robust to rare very-low-LVEF outliers
- Checkpoint saved when `val MAE` improves

### Task 2 — LV Dysfunction Classification

```bash
python train_task2.py
python train_task2.py --epochs 60 --focal_alpha 0.75 --focal_gamma 2.0
```

Key design choices:
- **Focal loss** (`alpha=0.75, gamma=2.0`) — handles 82/18 class imbalance without manual pos_weight tuning
- Checkpoint saved when `val AUC-ROC` improves

### Task 3 — Cardiotoxicity Prediction

```bash
# Recommended: load Task 1 backbone
python train_task3.py --pretrained ../checkpoints/task1/best.pth

# Or Task 2 backbone
python train_task3.py --pretrained ../checkpoints/task2/best.pth
```

Key design choices:
- **Frozen backbone**, only the MLP head is trained (~0.3% of total parameters)
- **Weighted BCE** (`pos_weight = neg/pos`) computed from training labels
- 30 epochs, higher LR (`1e-3`) since only the head trains
- Metrics: AUC-ROC, balanced accuracy, Brier score

---

## Data Format

Each DICOM is a grayscale echocardiography video:
- Colour space: `YBR_FULL_422` JPEG-compressed
- Shape: `(T, H, W, 3)` — Y channel (luma) extracted as grayscale
- Frame count: ~25–68 frames per clip
- Resolution: ~708 × 1016 px (resized to 112 × 112 during loading)

Frame sampling: `np.linspace(0, T-1, 32)` — uniform, always exactly 32 frames.

---

## Augmentation (training only)

- Random horizontal flip
- Random brightness / contrast jitter
- Temporal jitter: random start offset up to `T // 4`

---

## Label Columns

| Task | CSV column | Type |
|------|------------|------|
| 1 | `lvef` | float (%) |
| 2 | `lv_dysfunction` | 0 / 1 |
| 3 | `cardiotoxicity` | 0 / 1 |

---

## Hardware

Tested on 2× NVIDIA RTX 6000 Ada (49 GB each), driver 572.16, CUDA 12.x.  
Mixed-precision training (AMP) enabled automatically when a GPU is available.
