"""
Generate prediction CSVs for EchoRisk Tasks 1, 2, and 3 — EchoPrime backbone.

Usage — val split (default):
    python src_echoprime/predict.py --task 1 --checkpoint checkpoints_echoprime/task1/best.pth
    python src_echoprime/predict.py --task 2 --checkpoint checkpoints_echoprime/task2/best.pth
    python src_echoprime/predict.py --task 3 --checkpoint checkpoints_echoprime/task3/best.pth

Usage — test / Docker manifest:
    python src_echoprime/predict.py --task 1 --checkpoint checkpoints_echoprime/task1/best.pth \
        --csv manifest.csv --dicom_dir /data/dicom --no_split_subdir --out task1_predictions.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from dataset import _load_video, _read_dicom_gray, _sample_frames, _resize_frames, _to_tensor
from model import EchoNet, ECHOPRIME_ENCODER_WEIGHTS

_TASK_DEFAULTS = {
    1: dict(
        csv_val   = '/mnt/e/challenge/echo_risk/data/task1/Task1/public/labels/task1_val.csv',
        dicom     = '/mnt/e/challenge/echo_risk/data/task1/Task1/public/dicom',
        task_type = 'regression',
        pred_col  = 'lvef_pred',
        out       = 'task1_val_preds_echoprime.csv',
    ),
    2: dict(
        csv_val   = '/mnt/e/challenge/echo_risk/data/task2/Task2/public/labels/task2_val.csv',
        dicom     = '/mnt/e/challenge/echo_risk/data/task2/Task2/public/dicom',
        task_type = 'classification',
        pred_col  = 'dysfunction_pred',
        out       = 'task2_val_preds_echoprime.csv',
    ),
    3: dict(
        csv_val   = '/mnt/e/challenge/echo_risk/data/task3/Task3/public/labels/task3_val.csv',
        dicom     = '/mnt/e/challenge/echo_risk/data/task3/Task3/public/dicom',
        task_type = 'classification',
        pred_col  = 'cardiotoxicity_pred',
        out       = 'task3_val_preds_echoprime.csv',
    ),
}


class EchoPredictDataset(Dataset):
    def __init__(self, csv_path, dicom_dir, split, num_frames=16, img_size=224,
                 no_split_subdir=False):
        self.df              = pd.read_csv(csv_path)
        self.dicom_dir       = dicom_dir
        self.split           = split
        self.num_frames      = num_frames
        self.img_size        = img_size
        self.no_split_subdir = no_split_subdir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        patient_id = str(row['patient_id'])
        timepoint  = str(row['timepoint'])

        if self.no_split_subdir:
            def load(filename):
                if pd.isna(filename) or str(filename).strip() == '':
                    return torch.zeros(3, self.num_frames, self.img_size, self.img_size), False
                path = os.path.join(self.dicom_dir, patient_id, timepoint, str(filename))
                if not os.path.exists(path):
                    return torch.zeros(3, self.num_frames, self.img_size, self.img_size), False
                gray   = _read_dicom_gray(path)
                frames = _sample_frames(gray, self.num_frames, temporal_jitter=False)
                frames = _resize_frames(frames, self.img_size)
                return _to_tensor(frames), True
            a4c, has_a4c = load(row.get('video_a4c'))
            a2c, has_a2c = load(row.get('video_a2c'))
        else:
            a4c, has_a4c = _load_video(self.dicom_dir, self.split, patient_id, timepoint,
                                       row.get('video_a4c'), self.num_frames, self.img_size, augment=False)
            a2c, has_a2c = _load_video(self.dicom_dir, self.split, patient_id, timepoint,
                                       row.get('video_a2c'), self.num_frames, self.img_size, augment=False)

        if not has_a4c and has_a2c:
            a4c = a2c.clone()

        has_a2c_flag = torch.tensor(float(has_a2c), dtype=torch.float32)
        return a4c, a2c, has_a2c_flag, patient_id, timepoint


def parse_args():
    p = argparse.ArgumentParser(description='EchoRisk inference — EchoPrime backbone')
    p.add_argument('--task',            type=int, required=True, choices=[1, 2, 3])
    p.add_argument('--checkpoint',      required=True)
    p.add_argument('--encoder_weights', default=ECHOPRIME_ENCODER_WEIGHTS)
    p.add_argument('--csv',             default=None)
    p.add_argument('--split',           default='val')
    p.add_argument('--dicom_dir',       default=None)
    p.add_argument('--out',             default=None)
    p.add_argument('--num_frames',      type=int, default=16)
    p.add_argument('--img_size',        type=int, default=224)
    p.add_argument('--batch_size',      type=int, default=4)
    p.add_argument('--num_workers',     type=int, default=4)
    p.add_argument('--no_split_subdir', action='store_true')
    return p.parse_args()


def main():
    args  = parse_args()
    defs  = _TASK_DEFAULTS[args.task]
    csv   = args.csv      or defs['csv_val']
    dicom = args.dicom_dir or defs['dicom']
    out   = args.out      or defs['out']

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f'Task {args.task} | device: {device} | checkpoint: {args.checkpoint}')
    print(f'CSV: {csv}')

    freeze_backbone = (args.task == 3)
    model = EchoNet(task=defs['task_type'], freeze_backbone=freeze_backbone,
                    encoder_weights=args.encoder_weights).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'Loaded checkpoint (saved at epoch {ckpt.get("epoch", "?")})')

    ds = EchoPredictDataset(
        csv_path       = csv,
        dicom_dir      = dicom,
        split          = args.split,
        num_frames     = args.num_frames,
        img_size       = args.img_size,
        no_split_subdir= args.no_split_subdir,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=use_amp)
    print(f'Running inference on {len(ds)} exams...')

    pred_col = defs['pred_col']
    rows     = []
    with torch.no_grad():
        for a4c, a2c, has_a2c, patient_ids, timepoints in tqdm(loader, unit='batch', dynamic_ncols=True):
            a4c     = a4c.to(device, non_blocking=True)
            a2c     = a2c.to(device, non_blocking=True)
            has_a2c = has_a2c.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(a4c, a2c, has_a2c)
            if defs['task_type'] == 'regression':
                preds = logits.float().clamp(0, 100).cpu().numpy()
            else:
                preds = torch.sigmoid(logits.float()).cpu().numpy()
            for pid, tp, pred in zip(patient_ids, timepoints, preds):
                rows.append({'patient_id': pid, 'timepoint': tp, pred_col: float(pred)})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out, index=False)
    print(f'Saved {len(df_out)} predictions → {out}')

    # Auto-score when running against val
    gt_csv = defs['csv_val']
    if csv == gt_csv and os.path.exists(gt_csv):
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'echorisk-benchmark', 'evaluation', 'src'))
            from echorisk_eval.metrics import score_task1, score_task2, score_task3
            gt = pd.read_csv(gt_csv)
            if args.task == 1:
                r = score_task1(df_out, gt)
                print(f'\nVal — MAE: {r.mae:.4f} | RMSE: {r.rmse:.4f} | Pearson r: {r.pearson_r:.4f} | n={r.n}')
            elif args.task == 2:
                r = score_task2(df_out, gt)
                print(f'\nVal — AUC: {r.auroc:.4f} | BalAcc: {r.balanced_accuracy:.4f} | Sens@90Spec: {r.sensitivity_at_90_specificity:.4f} | n={r.n}')
            else:
                r = score_task3(df_out, gt)
                print(f'\nVal — AUC: {r.auroc:.4f} | BalAcc: {r.balanced_accuracy:.4f} | Sens@FPR10: {r.sensitivity_at_fpr_10:.4f} | Brier: {r.brier:.4f} | n={r.n}')
        except ImportError:
            print('\n(echorisk-benchmark not installed — skipping local score)')


if __name__ == '__main__':
    main()
