"""
Task 3 — Early Prediction of Cardiotoxicity (EchoPrime backbone)

Metric:  AUC-ROC (higher is better).
Strategy:
  1. Load pretrained EchoPrime backbone from Task 2 checkpoint (classification transfer).
  2. Unfreeze last N MViT blocks with a low backbone LR; train head at higher LR.
  3. Weighted BCE (pos_weight from training label distribution).
  4. Strong augmentation (rotation, crop, noise, reversal) for the small dataset.

Usage:
    python src_echoprime/train_task3.py
    python src_echoprime/train_task3.py --resume   # continue from last.pth
    python src_echoprime/train_task3.py --pretrained checkpoints_echoprime/task1/best.pth
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
    p = argparse.ArgumentParser(description='EchoRisk Task 3 — Cardiotoxicity (EchoPrime)')
    p.add_argument('--data_root',        default='/mnt/e/challenge/echo_risk/data/task3/Task3/public')
    p.add_argument('--pretrained',       default='/mnt/e/challenge/echo_risk/checkpoints_echoprime/task2/best.pth',
                   help='Task 1 or Task 2 checkpoint for backbone initialisation')
    p.add_argument('--encoder_weights',  default=ECHOPRIME_ENCODER_WEIGHTS)
    p.add_argument('--checkpoint_dir',   default='/mnt/e/challenge/echo_risk/checkpoints_echoprime/task3')
    p.add_argument('--epochs',           type=int,   default=40)
    p.add_argument('--batch_size',       type=int,   default=4)
    p.add_argument('--lr',               type=float, default=1e-4,
                   help='Head LR')
    p.add_argument('--backbone_lr',      type=float, default=1e-5,
                   help='LR for unfrozen backbone blocks')
    p.add_argument('--unfreeze_blocks',  type=int,   default=4,
                   help='Number of final MViT blocks to unfreeze (0 = head-only)')
    p.add_argument('--weight_decay',     type=float, default=1e-3)
    p.add_argument('--num_frames',       type=int,   default=16)
    p.add_argument('--img_size',         type=int,   default=224)
    p.add_argument('--num_workers',      type=int,   default=4)
    p.add_argument('--seed',             type=int,   default=42)
    p.add_argument('--resume',           action='store_true',
                   help='Resume training from last.pth in checkpoint_dir')
    return p.parse_args()


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_backbone(model: EchoNet, ckpt_path: str, device: torch.device):
    """Load only backbone weights from a Task 1/2 checkpoint, ignore head."""
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f'Warning: pretrained checkpoint not found at "{ckpt_path}". Training from EchoPrime init only.')
        return
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    saved = ckpt['model']
    current = model.state_dict()
    backbone_weights = {k: v for k, v in saved.items() if k.startswith('backbone.')}
    current.update(backbone_weights)
    model.load_state_dict(current)
    print(f'Loaded {len(backbone_weights)} backbone tensors from {ckpt_path}')


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
        pbar.set_postfix(loss=f'{loss.item():.4f}', avg=f'{total_loss/i:.4f}')
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    for a4c, a2c, has_a2c, labels in tqdm(loader, desc='  Val  ', unit='batch', leave=False, dynamic_ncols=True):
        a4c, a2c, has_a2c, labels_dev = (
            a4c.to(device, non_blocking=True),
            a2c.to(device, non_blocking=True),
            has_a2c.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(a4c, a2c, has_a2c)
            loss   = criterion(logits, labels_dev)
        total_loss += loss.item()
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())
    probs, labels = np.array(all_probs), np.array(all_labels)
    auc      = roc_auc_score(labels, probs)
    bal_acc  = balanced_accuracy_score(labels, (probs >= 0.5).astype(int))
    brier    = brier_score_loss(labels, probs)
    val_loss = total_loss / len(loader)
    return auc, bal_acc, brier, val_loss, probs, labels


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    sys.stdout = _Tee(os.path.join(args.checkpoint_dir, 'train.log'))

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'

    print('=' * 60)
    print('Task 3 — Cardiotoxicity Prediction (EchoPrime backbone)')
    print('=' * 60)
    print(f'Device: {device} | AMP: {use_amp}')
    print(f'Epochs: {args.epochs} | Batch: {args.batch_size} | Head LR: {args.lr} | Backbone LR: {args.backbone_lr}')
    print(f'Unfreeze last {args.unfreeze_blocks} backbone blocks')
    print(f'Frames: {args.num_frames} | Img size: {args.img_size}')
    print(f'Pretrained: {args.pretrained or "none"}')
    print('-' * 60)

    train_labels_df = pd.read_csv(
        os.path.join(args.data_root, 'labels/task3_train.csv')
    )['cardiotoxicity']
    n_neg = int((train_labels_df == 0).sum())
    n_pos = int((train_labels_df == 1).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    print(f'Class dist: pos={n_pos}, neg={n_neg} | pos_weight={pos_weight.item():.3f}')

    train_ds = EchoDataset(
        csv_path       = os.path.join(args.data_root, 'labels/task3_train.csv'),
        dicom_dir      = os.path.join(args.data_root, 'dicom'),
        split          = 'train',
        label_col      = 'cardiotoxicity',
        num_frames     = args.num_frames,
        img_size       = args.img_size,
        augment        = True,
        strong_augment = True,
    )
    val_ds = EchoDataset(
        csv_path   = os.path.join(args.data_root, 'labels/task3_val.csv'),
        dicom_dir  = os.path.join(args.data_root, 'dicom'),
        split      = 'val',
        label_col  = 'cardiotoxicity',
        num_frames = args.num_frames,
        img_size   = args.img_size,
        augment    = False,
    )
    print(f'Train: {len(train_ds)} | Val: {len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=use_amp, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=use_amp)

    # Build with frozen backbone, then selectively unfreeze last N blocks
    model = EchoNet(task='classification', freeze_backbone=True,
                    encoder_weights=args.encoder_weights).to(device)

    # Load backbone from Task 1/2 only when starting fresh (not resuming)
    if not args.resume:
        load_backbone(model, args.pretrained, device)

    # Unfreeze the last `unfreeze_blocks` transformer blocks of MViT backbone
    if args.unfreeze_blocks > 0:
        all_blocks = list(model.backbone.blocks)
        for blk in all_blocks[-args.unfreeze_blocks:]:
            for p in blk.parameters():
                p.requires_grad_(True)

    backbone_params = [p for name, p in model.named_parameters()
                       if p.requires_grad and name.startswith('backbone.')]
    head_params     = [p for name, p in model.named_parameters()
                       if p.requires_grad and not name.startswith('backbone.')]

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)')
    print(f'  backbone unfrozen: {sum(p.numel() for p in backbone_params):,} | head: {sum(p.numel() for p in head_params):,}')
    print('-' * 60)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [
            {'params': backbone_params, 'lr': args.backbone_lr},
            {'params': head_params,     'lr': args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler    = torch.amp.GradScaler('cuda') if use_amp else None

    start_epoch = 1
    best_auc    = 0.0

    last_path = os.path.join(args.checkpoint_dir, 'last.pth')
    if args.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        if scaler and 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_auc    = ckpt.get('best_auc', 0.0)
        print(f'Resumed from epoch {ckpt["epoch"]} | best AUC so far: {best_auc:.4f}')
    elif args.resume:
        print(f'No last.pth found at {last_path} — starting from scratch with pretrained backbone.')
        load_backbone(model, args.pretrained, device)

    for epoch in range(start_epoch, args.epochs + 1):
        t0         = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, use_amp, epoch, args.epochs)
        val_auc, val_bal_acc, val_brier, val_loss, _, _ = evaluate(model, val_loader, criterion, device, use_amp)
        scheduler.step()

        print(
            f'Epoch {epoch:3d}/{args.epochs} | '
            f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
            f'AUC: {val_auc:.4f} | BalAcc: {val_bal_acc:.4f} | Brier: {val_brier:.4f} | '
            f'LR: {scheduler.get_last_lr()[0]:.2e} | {time.time()-t0:.0f}s'
        )

        last_state = {
            'epoch': epoch, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'best_auc': best_auc, 'auc': val_auc,
        }
        if scaler:
            last_state['scaler'] = scaler.state_dict()
        torch.save(last_state, last_path)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'auc': val_auc, 'bal_acc': val_bal_acc, 'brier': val_brier},
                       os.path.join(args.checkpoint_dir, 'best.pth'))
            print(f'  → Saved new best (AUC={val_auc:.4f})')

    print(f'\nFinished. Best Val AUC: {best_auc:.4f}')


if __name__ == '__main__':
    main()
