"""
Task 3 — Early Prediction of Cardiotoxicity (baseline T1 only)

Metric:  AUC-ROC (higher is better).
Dataset: Only 157 training patients — very small. Overfitting is the main risk.

Strategy:
  1. Load a pretrained backbone from Task 1 (or Task 2) checkpoint.
  2. Freeze the backbone entirely — only the MLP head is trained.
  3. Use weighted BCE (pos_weight computed from training label distribution).
  4. Short training (30 epochs), higher LR for the head, cosine decay.

This avoids catastrophic forgetting and dramatically reduces the effective
parameter count, which is critical given the tiny training set.

Usage:
    # First train Task 1 or Task 2, then:
    python train_task3.py --pretrained ../checkpoints/task1/best.pth
    python train_task3.py --pretrained ../checkpoints/task2/best.pth
    # Or train from scratch (not recommended):
    python train_task3.py --pretrained ''
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, brier_score_loss
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from dataset import EchoDataset
from model import EchoNet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='EchoRisk Task 3 — Cardiotoxicity Prediction')
    p.add_argument('--data_root',       default='/mnt/e/challenge/echo_risk/data/task3/Task3/public')
    p.add_argument('--pretrained',      default='/mnt/e/challenge/echo_risk/checkpoints/task1/best.pth',
                   help='Path to Task 1 / Task 2 checkpoint for backbone initialisation')
    p.add_argument('--checkpoint_dir',  default='/mnt/e/challenge/echo_risk/checkpoints/task3')
    p.add_argument('--epochs',          type=int,   default=30)
    p.add_argument('--batch_size',      type=int,   default=4)
    p.add_argument('--lr',              type=float, default=1e-3,
                   help='Higher than Tasks 1/2 because only the head trains')
    p.add_argument('--weight_decay',    type=float, default=1e-3)
    p.add_argument('--num_frames',      type=int,   default=32)
    p.add_argument('--img_size',        type=int,   default=112)
    p.add_argument('--num_workers',     type=int,   default=4)
    p.add_argument('--seed',            type=int,   default=42)
    return p.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_backbone(model: EchoNet, ckpt_path: str, device: torch.device):
    """Load only backbone weights from a Task 1/2 checkpoint, ignore head."""
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f'Warning: pretrained checkpoint not found at "{ckpt_path}". Training from scratch.')
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    saved = ckpt['model']

    # Extract backbone keys only — head shapes differ between tasks
    current = model.state_dict()
    backbone_weights = {k: v for k, v in saved.items() if k.startswith('backbone.')}
    current.update(backbone_weights)
    model.load_state_dict(current)

    n_loaded = len(backbone_weights)
    print(f'Loaded {n_loaded} backbone tensors from {ckpt_path}')


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, criterion, device, use_amp, epoch, epochs):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f'  Train {epoch}/{epochs}', unit='batch', leave=False,
                dynamic_ncols=True)
    for batch_idx, (a4c, a2c, has_a2c, labels) in enumerate(pbar):
        a4c     = a4c.to(device, non_blocking=True)
        a2c     = a2c.to(device, non_blocking=True)
        has_a2c = has_a2c.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(a4c, a2c, has_a2c)
            loss   = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        avg = total_loss / (batch_idx + 1)
        pbar.set_postfix(loss=f'{loss.item():.4f}', avg=f'{avg:.4f}')

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()
    all_probs, all_labels = [], []

    pbar = tqdm(loader, desc='  Val  ', unit='batch', leave=False, dynamic_ncols=True)
    for a4c, a2c, has_a2c, labels in pbar:
        a4c     = a4c.to(device, non_blocking=True)
        a2c     = a2c.to(device, non_blocking=True)
        has_a2c = has_a2c.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(a4c, a2c, has_a2c)

        probs = torch.sigmoid(logits.float()).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())

    probs  = np.array(all_probs)
    labels = np.array(all_labels)

    auc      = roc_auc_score(labels, probs)
    bal_acc  = balanced_accuracy_score(labels, (probs >= 0.5).astype(int))
    brier    = brier_score_loss(labels, probs)
    return auc, bal_acc, brier, probs, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print('=' * 60)
    print('Task 3 — Cardiotoxicity Prediction (transfer learning)')
    print('=' * 60)
    print(f'Device      : {device} | AMP: {use_amp}')
    print(f'Epochs      : {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}')
    print(f'Frames      : {args.num_frames} | Img size: {args.img_size}')
    print(f'Pretrained  : {args.pretrained or "none (scratch)"}')
    print(f'Data root   : {args.data_root}')
    print(f'Checkpoints : {args.checkpoint_dir}')
    print('-' * 60)

    # ---- Compute pos_weight from training distribution ----------------------
    train_labels_df = pd.read_csv(
        os.path.join(args.data_root, 'labels/task3_train.csv')
    )['cardiotoxicity']
    n_neg = int((train_labels_df == 0).sum())
    n_pos = int((train_labels_df == 1).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    print(f'Class dist  : pos={n_pos}, neg={n_neg} | pos_weight={pos_weight.item():.3f}', flush=True)

    # ---- Data ---------------------------------------------------------------
    print('Loading datasets...', flush=True)
    train_ds = EchoDataset(
        csv_path  = os.path.join(args.data_root, 'labels/task3_train.csv'),
        dicom_dir = os.path.join(args.data_root, 'dicom'),
        split     = 'train',
        label_col = 'cardiotoxicity',
        num_frames = args.num_frames,
        img_size   = args.img_size,
        augment    = True,
    )
    val_ds = EchoDataset(
        csv_path  = os.path.join(args.data_root, 'labels/task3_val.csv'),
        dicom_dir = os.path.join(args.data_root, 'dicom'),
        split     = 'val',
        label_col = 'cardiotoxicity',
        num_frames = args.num_frames,
        img_size   = args.img_size,
        augment    = False,
    )
    print(f'Train: {len(train_ds)} patients | Val: {len(val_ds)} patients', flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=use_amp, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=use_amp,
    )

    # ---- Model — frozen backbone, trainable head only -----------------------
    model = EchoNet(task='classification', freeze_backbone=True).to(device)
    load_backbone(model, args.pretrained, device)

    # Verify: backbone is frozen, head is trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f'Parameters  : {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)', flush=True)
    print('-' * 60, flush=True)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # ---- Training loop ------------------------------------------------------
    best_auc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device, use_amp,
            epoch, args.epochs
        )
        val_auc, val_bal_acc, val_brier, _, _ = evaluate(
            model, val_loader, device, use_amp
        )
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f'Epoch {epoch:3d}/{args.epochs} | '
            f'Train Loss: {train_loss:.4f} | '
            f'Val AUC: {val_auc:.4f} | '
            f'Val BalAcc: {val_bal_acc:.4f} | '
            f'Val Brier: {val_brier:.4f} | '
            f'LR: {scheduler.get_last_lr()[0]:.2e} | '
            f'{elapsed:.0f}s'
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {'epoch': epoch, 'model': model.state_dict(),
                 'auc': val_auc, 'bal_acc': val_bal_acc, 'brier': val_brier},
                os.path.join(args.checkpoint_dir, 'best.pth')
            )
            print(f'  → Saved new best (AUC={val_auc:.4f})')

    print(f'\nFinished. Best Val AUC: {best_auc:.4f}')


if __name__ == '__main__':
    main()
