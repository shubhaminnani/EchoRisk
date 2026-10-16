"""
Task 1 — LVEF Regression

Metric:  MAE (lower is better). Clinical target ≤ 5%.
Loss:    Huber loss (delta=5.0 %) — less sensitive to the rare very-low-LVEF outliers
         than MSE, while still penalising large errors more than MAE alone.
Output:  Raw scalar clamped to [0, 100] at inference time.

Usage:
    python train_task1.py
    python train_task1.py --epochs 80 --batch_size 8 --lr 5e-5
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from dataset import EchoDataset
from model import EchoNet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='EchoRisk Task 1 — LVEF Regression')
    p.add_argument('--data_root',       default='/mnt/e/challenge/echo_risk/data/task1/Task1/public')
    p.add_argument('--checkpoint_dir',  default='/mnt/e/challenge/echo_risk/checkpoints/task1')
    p.add_argument('--epochs',          type=int,   default=50)
    p.add_argument('--batch_size',      type=int,   default=4)
    p.add_argument('--lr',              type=float, default=1e-4)
    p.add_argument('--weight_decay',    type=float, default=1e-4)
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


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, criterion, device, use_amp, epoch, epochs):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f'  Train {epoch}/{epochs}', unit='batch', leave=False,
                dynamic_ncols=True)
    for batch_idx, (a4c, a2c, has_a2c, labels) in enumerate(pbar):
        a4c      = a4c.to(device, non_blocking=True)
        a2c      = a2c.to(device, non_blocking=True)
        has_a2c  = has_a2c.to(device, non_blocking=True)
        labels   = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            preds = model(a4c, a2c, has_a2c)
            loss  = criterion(preds, labels)

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
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc='  Val  ', unit='batch', leave=False, dynamic_ncols=True)
    for a4c, a2c, has_a2c, labels in pbar:
        a4c     = a4c.to(device, non_blocking=True)
        a2c     = a2c.to(device, non_blocking=True)
        has_a2c = has_a2c.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            preds = model(a4c, a2c, has_a2c)

        preds = preds.float().clamp(0.0, 100.0).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    mae  = mean_absolute_error(all_labels, all_preds)
    rmse = float(np.sqrt(np.mean((np.array(all_preds) - np.array(all_labels)) ** 2)))
    return mae, rmse, np.array(all_preds), np.array(all_labels)


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
    print('Task 1 — LVEF Regression')
    print('=' * 60)
    print(f'Device      : {device} | AMP: {use_amp}')
    print(f'Epochs      : {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}')
    print(f'Frames      : {args.num_frames} | Img size: {args.img_size}')
    print(f'Data root   : {args.data_root}')
    print(f'Checkpoints : {args.checkpoint_dir}')
    print('-' * 60)

    # ---- Data ---------------------------------------------------------------
    print('Loading datasets...', flush=True)
    train_ds = EchoDataset(
        csv_path  = os.path.join(args.data_root, 'labels/task1_train.csv'),
        dicom_dir = os.path.join(args.data_root, 'dicom'),
        split     = 'train',
        label_col = 'lvef',
        num_frames = args.num_frames,
        img_size   = args.img_size,
        augment    = True,
    )
    val_ds = EchoDataset(
        csv_path  = os.path.join(args.data_root, 'labels/task1_val.csv'),
        dicom_dir = os.path.join(args.data_root, 'dicom'),
        split     = 'val',
        label_col = 'lvef',
        num_frames = args.num_frames,
        img_size   = args.img_size,
        augment    = False,
    )
    print(f'Train: {len(train_ds)} exams | Val: {len(val_ds)} exams', flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=use_amp, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=use_amp,
    )

    # ---- Model / optimiser --------------------------------------------------
    print('Building model...', flush=True)
    model     = EchoNet(task='regression').to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters  : {total_params:,}', flush=True)
    print('-' * 60, flush=True)
    criterion = nn.HuberLoss(delta=5.0)   # delta = clinically meaningful error
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # ---- Training loop ------------------------------------------------------
    best_mae = float('inf')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device, use_amp,
            epoch, args.epochs
        )
        val_mae, val_rmse, _, _ = evaluate(model, val_loader, device, use_amp)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f'Epoch {epoch:3d}/{args.epochs} | '
            f'Train Loss: {train_loss:.4f} | '
            f'Val MAE: {val_mae:.2f}% | '
            f'Val RMSE: {val_rmse:.2f}% | '
            f'LR: {scheduler.get_last_lr()[0]:.2e} | '
            f'{elapsed:.0f}s'
        )

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(
                {'epoch': epoch, 'model': model.state_dict(),
                 'mae': val_mae, 'rmse': val_rmse},
                os.path.join(args.checkpoint_dir, 'best.pth')
            )
            print(f'  → Saved new best (MAE={val_mae:.2f}%)')

    print(f'\nFinished. Best Val MAE: {best_mae:.2f}%')


if __name__ == '__main__':
    main()
