"""
Task 1 — LVEF Estimation (EchoPrime backbone)

Metric:  MAE (lower is better). Clinical target: MAE ≤ 5%.
Backbone: MViT-v2-S pretrained on echocardiography (EchoPrime weights).

Usage:
    python src_echoprime/train_task1.py
    python src_echoprime/train_task1.py --resume   # continue from last.pth
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
from scipy.stats import pearsonr
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from dataset import EchoDataset
from model import EchoNet, ECHOPRIME_ENCODER_WEIGHTS


class _Tee:
    """Write stdout to both console and a log file."""
    def __init__(self, path):
        self._term = sys.stdout
        self._file = open(path, 'a')
    def write(self, msg):
        self._term.write(msg)
        self._file.write(msg)
        self._file.flush()
    def flush(self):
        self._term.flush()
        self._file.flush()


def parse_args():
    p = argparse.ArgumentParser(description='EchoRisk Task 1 — LVEF (EchoPrime)')
    p.add_argument('--data_root',       default='/mnt/e/challenge/echo_risk/data/task1/Task1/public')
    p.add_argument('--encoder_weights', default=ECHOPRIME_ENCODER_WEIGHTS)
    p.add_argument('--checkpoint_dir',  default='/mnt/e/challenge/echo_risk/checkpoints_echoprime/task1')
    p.add_argument('--epochs',          type=int,   default=50)
    p.add_argument('--batch_size',      type=int,   default=4)
    p.add_argument('--lr',              type=float, default=1e-4)
    p.add_argument('--weight_decay',    type=float, default=1e-4)
    p.add_argument('--num_frames',      type=int,   default=16)
    p.add_argument('--img_size',        type=int,   default=224)
    p.add_argument('--num_workers',     type=int,   default=4)
    p.add_argument('--seed',            type=int,   default=42)
    p.add_argument('--resume',          action='store_true',
                   help='Resume training from last.pth in checkpoint_dir')
    return p.parse_args()


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, scaler, criterion, device, use_amp, epoch, epochs):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f'  Train {epoch}/{epochs}', unit='batch', leave=False, dynamic_ncols=True)
    for i, (a4c, a2c, has_a2c, labels) in enumerate(pbar, 1):
        a4c, a2c, has_a2c, labels = (
            a4c.to(device, non_blocking=True),
            a2c.to(device, non_blocking=True),
            has_a2c.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )
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
        pbar.set_postfix(loss=f'{loss.item():.4f}', avg=f'{total_loss/i:.4f}')
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    for a4c, a2c, has_a2c, labels in tqdm(loader, desc='  Val  ', unit='batch', leave=False, dynamic_ncols=True):
        a4c, a2c, has_a2c, labels_dev = (
            a4c.to(device, non_blocking=True),
            a2c.to(device, non_blocking=True),
            has_a2c.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )
        with torch.autocast(device_type=device.type, enabled=use_amp):
            preds = model(a4c, a2c, has_a2c)
            loss  = criterion(preds, labels_dev)
        total_loss += loss.item()
        preds = preds.float().clamp(0, 100).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    p, l     = np.array(all_preds), np.array(all_labels)
    mae      = mean_absolute_error(l, p)
    rmse     = float(np.sqrt(np.mean((p - l) ** 2)))
    r        = float(pearsonr(p, l).statistic) if np.std(p) > 0 else float('nan')
    val_loss = total_loss / len(loader)
    return mae, rmse, r, val_loss, p, l


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    sys.stdout = _Tee(os.path.join(args.checkpoint_dir, 'train.log'))

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'

    print('=' * 60)
    print('Task 1 — LVEF Estimation (EchoPrime backbone)')
    print('=' * 60)
    print(f'Device: {device} | AMP: {use_amp}')
    print(f'Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}')
    print(f'Frames: {args.num_frames} | Img size: {args.img_size}')
    print(f'Encoder weights: {args.encoder_weights}')
    print('-' * 60)

    train_ds = EchoDataset(
        csv_path       = os.path.join(args.data_root, 'labels/task1_train.csv'),
        dicom_dir      = os.path.join(args.data_root, 'dicom'),
        split          = 'train',
        label_col      = 'lvef',
        num_frames     = args.num_frames,
        img_size       = args.img_size,
        augment        = True,
        strong_augment = True,
    )
    val_ds = EchoDataset(
        csv_path   = os.path.join(args.data_root, 'labels/task1_val.csv'),
        dicom_dir  = os.path.join(args.data_root, 'dicom'),
        split      = 'val',
        label_col  = 'lvef',
        num_frames = args.num_frames,
        img_size   = args.img_size,
        augment    = False,
    )
    print(f'Train: {len(train_ds)} | Val: {len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=use_amp, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=use_amp)

    model = EchoNet(task='regression', freeze_backbone=False,
                    encoder_weights=args.encoder_weights).to(device)

    criterion = nn.HuberLoss(delta=5.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler    = torch.amp.GradScaler('cuda') if use_amp else None

    start_epoch = 1
    best_mae    = float('inf')

    last_path = os.path.join(args.checkpoint_dir, 'last.pth')
    if args.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        if scaler and 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_mae    = ckpt.get('best_mae', float('inf'))
        print(f'Resumed from epoch {ckpt["epoch"]} | best MAE so far: {best_mae:.4f}')
    elif args.resume:
        print(f'No last.pth found at {last_path} — starting from scratch.')

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)')
    print('-' * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        t0         = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, use_amp, epoch, args.epochs)
        val_mae, val_rmse, val_r, val_loss, _, _ = evaluate(model, val_loader, criterion, device, use_amp)
        scheduler.step()

        print(
            f'Epoch {epoch:3d}/{args.epochs} | '
            f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
            f'MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f} | r: {val_r:.4f} | '
            f'LR: {scheduler.get_last_lr()[0]:.2e} | {time.time()-t0:.0f}s'
        )

        # Save full state for resume every epoch
        last_state = {
            'epoch': epoch, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'best_mae': best_mae, 'mae': val_mae,
        }
        if scaler:
            last_state['scaler'] = scaler.state_dict()
        torch.save(last_state, last_path)

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'mae': val_mae, 'rmse': val_rmse, 'pearson_r': val_r},
                       os.path.join(args.checkpoint_dir, 'best.pth'))
            print(f'  → Saved new best (MAE={val_mae:.4f})')

    print(f'\nFinished. Best Val MAE: {best_mae:.4f}')


if __name__ == '__main__':
    main()
