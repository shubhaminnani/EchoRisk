"""
Task 2 — LV Dysfunction Classification

Metric:  AUC-ROC (higher is better).
Loss:    Focal loss — handles the 82/18 class imbalance without requiring
         manual pos_weight tuning. alpha=0.75 up-weights the positive class;
         gamma=2.0 down-weights easy negatives.
Output:  Raw logit → sigmoid for probability in [0, 1].

Usage:
    python train_task2.py 
    python train_task2.py --epochs 60 --batch_size 4
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from dataset import EchoDataset
from model import EchoNet


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Binary focal loss for class-imbalanced problems.
    alpha: weight for the positive class (set > 0.5 to up-weight the minority).
    gamma: focusing parameter; higher = more focus on hard examples.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt  = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - pt) ** self.gamma * bce
        return loss.mean()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='EchoRisk Task 2 — LV Dysfunction')
    p.add_argument('--data_root',       default='/mnt/e/challenge/echo_risk/data/task2/Task2/public')
    p.add_argument('--checkpoint_dir',  default='/mnt/e/challenge/echo_risk/checkpoints/task2')
    p.add_argument('--epochs',          type=int,   default=50)
    p.add_argument('--batch_size',      type=int,   default=4)
    p.add_argument('--lr',              type=float, default=1e-4)
    p.add_argument('--weight_decay',    type=float, default=1e-4)
    p.add_argument('--focal_alpha',     type=float, default=0.75)
    p.add_argument('--focal_gamma',     type=float, default=2.0)
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
    auc    = roc_auc_score(labels, probs)
    bal_acc = balanced_accuracy_score(labels, (probs >= 0.5).astype(int))
    return auc, bal_acc, probs, labels


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
    print('Task 2 — LV Dysfunction Classification')
    print('=' * 60)
    print(f'Device      : {device} | AMP: {use_amp}')
    print(f'Epochs      : {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}')
    print(f'Focal loss  : alpha={args.focal_alpha}, gamma={args.focal_gamma}')
    print(f'Frames      : {args.num_frames} | Img size: {args.img_size}')
    print(f'Data root   : {args.data_root}')
    print(f'Checkpoints : {args.checkpoint_dir}')
    print('-' * 60)

    # ---- Data ---------------------------------------------------------------
    print('Loading datasets...', flush=True)
    train_ds = EchoDataset(
        csv_path  = os.path.join(args.data_root, 'labels/task2_train.csv'),
        dicom_dir = os.path.join(args.data_root, 'dicom'),
        split     = 'train',
        label_col = 'lv_dysfunction',
        num_frames = args.num_frames,
        img_size   = args.img_size,
        augment    = True,
    )
    val_ds = EchoDataset(
        csv_path  = os.path.join(args.data_root, 'labels/task2_val.csv'),
        dicom_dir = os.path.join(args.data_root, 'dicom'),
        split     = 'val',
        label_col = 'lv_dysfunction',
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
    model     = EchoNet(task='classification').to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters  : {total_params:,}', flush=True)
    print('-' * 60, flush=True)
    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
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
        val_auc, val_bal_acc, _, _ = evaluate(model, val_loader, device, use_amp)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f'Epoch {epoch:3d}/{args.epochs} | '
            f'Train Loss: {train_loss:.4f} | '
            f'Val AUC: {val_auc:.4f} | '
            f'Val BalAcc: {val_bal_acc:.4f} | '
            f'LR: {scheduler.get_last_lr()[0]:.2e} | '
            f'{elapsed:.0f}s'
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {'epoch': epoch, 'model': model.state_dict(),
                 'auc': val_auc, 'bal_acc': val_bal_acc},
                os.path.join(args.checkpoint_dir, 'best.pth')
            )
            print(f'  → Saved new best (AUC={val_auc:.4f})')

    print(f'\nFinished. Best Val AUC: {best_auc:.4f}')


if __name__ == '__main__':
    main()
