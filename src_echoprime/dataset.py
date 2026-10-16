"""
Dataset utilities for EchoPrime backbone.

Key differences from baseline src/dataset.py:
  - 16 frames (EchoPrime uses stride-2 sampling from ~32 available frames)
  - 224×224 spatial resolution
  - Pixels kept in [0, 255] float range (not divided by 255)
  - 3-channel output tensor, normalised with EchoPrime per-channel stats
"""

import os
import random
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset
import cv2

# EchoPrime normalisation stats (raw pixel scale 0-255)
_MEAN = torch.tensor([29.110628, 28.076836, 29.096405])  # (3,)
_STD  = torch.tensor([47.989223, 46.456997, 47.20083])   # (3,)


def _read_dicom_gray(path: str) -> np.ndarray:
    """Return (T, H, W) float32 in [0, 255] from a DICOM cine loop."""
    ds  = pydicom.dcmread(path)
    arr = ds.pixel_array  # (T, H, W, 3) YBR or (T, H, W)
    if arr.ndim == 4:
        gray = arr[..., 0].astype(np.float32)   # Y luma channel
    else:
        gray = arr.astype(np.float32)
    return gray  # keep in [0, 255]


def _sample_frames(arr: np.ndarray, n: int, temporal_jitter: bool = False) -> np.ndarray:
    """Uniformly sample n frames from (T, H, W) with optional random start offset."""
    T = arr.shape[0]
    if temporal_jitter and T > n:
        max_offset = max(0, T // 4)
        start = random.randint(0, max_offset)
    else:
        start = 0
    indices = np.linspace(start, T - 1, n).round().astype(int)
    indices = np.clip(indices, 0, T - 1)
    return arr[indices]


def _resize_frames(frames: np.ndarray, size: int) -> np.ndarray:
    """Resize (T, H, W) to (T, size, size) using bilinear interpolation."""
    if frames.shape[1] == size and frames.shape[2] == size:
        return frames
    return np.stack([
        cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR)
        for f in frames
    ])


def _augment_spatial(frames: np.ndarray) -> np.ndarray:
    """Random horizontal flip + brightness/contrast jitter on (T, H, W) in [0,255]."""
    if random.random() < 0.5:
        frames = frames[:, :, ::-1].copy()
    if random.random() < 0.5:
        alpha = random.uniform(0.8, 1.2)
        beta  = random.uniform(-10.0, 10.0)   # pixel-scale offset
        frames = np.clip(alpha * frames + beta, 0.0, 255.0)
    return frames


def _augment_strong(frames: np.ndarray) -> np.ndarray:
    """Stronger augmentation for small datasets (Task 3). (T, H, W) in [0,255]."""
    T, H, W = frames.shape

    # Horizontal flip
    if random.random() < 0.5:
        frames = frames[:, :, ::-1].copy()

    # Brightness / contrast
    alpha = random.uniform(0.75, 1.25)
    beta  = random.uniform(-15.0, 15.0)
    frames = np.clip(alpha * frames + beta, 0.0, 255.0)

    # Random rotation ±12°
    if random.random() < 0.5:
        angle = random.uniform(-12, 12)
        M     = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
        frames = np.stack([
            cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REFLECT_101)
            for f in frames
        ])

    # Random zoom-crop: crop to 80-100% then resize back
    if random.random() < 0.5:
        scale = random.uniform(0.80, 1.0)
        ch, cw = int(H * scale), int(W * scale)
        y0 = random.randint(0, H - ch)
        x0 = random.randint(0, W - cw)
        frames = np.stack([
            cv2.resize(f[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_LINEAR)
            for f in frames
        ])

    # Gaussian noise (simulates ultrasound speckle)
    if random.random() < 0.5:
        noise  = np.random.normal(0, random.uniform(3.0, 8.0), frames.shape).astype(np.float32)
        frames = np.clip(frames + noise, 0.0, 255.0)

    # Temporal reversal (cardiac cycle is valid both directions)
    if random.random() < 0.3:
        frames = frames[::-1].copy()

    return frames


def _to_tensor(frames: np.ndarray) -> torch.Tensor:
    """(T, H, W) float32 in [0,255] → (3, T, H, W) EchoPrime-normalised tensor."""
    t = torch.from_numpy(np.ascontiguousarray(frames))     # (T, H, W)
    t = t.unsqueeze(0).expand(3, -1, -1, -1).float()      # (3, T, H, W)
    # Normalise per channel: (C, T, H, W) - mean/std shape (3,1,1,1)
    mean = _MEAN.view(3, 1, 1, 1)
    std  = _STD.view(3, 1, 1, 1)
    return (t - mean) / std


def _load_video(dicom_dir: str, split: str, patient_id: str,
                timepoint: str, filename, num_frames: int, img_size: int,
                augment: bool, strong_augment: bool = False) -> tuple[torch.Tensor, bool]:
    """
    Load a single view video. Returns (tensor (3, T, H, W), available: bool).
    Returns a zero tensor when the file is absent or missing.
    strong_augment: use heavier augmentation (rotation, crop, noise, reversal).
    """
    if pd.isna(filename) or str(filename).strip() == '':
        return torch.zeros(3, num_frames, img_size, img_size), False

    path = os.path.join(dicom_dir, split, patient_id, timepoint, str(filename))
    if not os.path.exists(path):
        return torch.zeros(3, num_frames, img_size, img_size), False

    gray   = _read_dicom_gray(path)
    frames = _sample_frames(gray, num_frames, temporal_jitter=augment)
    frames = _resize_frames(frames, img_size)
    if augment:
        frames = _augment_strong(frames) if strong_augment else _augment_spatial(frames)
    return _to_tensor(frames), True


class EchoDataset(Dataset):
    """
    Dataset for all three EchoRisk tasks — EchoPrime backbone variant.

    Outputs (3, T, H, W) tensors normalised to EchoPrime stats.
    Default: 16 frames, 224×224.
    """

    def __init__(self, csv_path: str, dicom_dir: str, split: str,
                 label_col: str, num_frames: int = 16,
                 img_size: int = 224, augment: bool = False,
                 strong_augment: bool = False):
        self.df             = pd.read_csv(csv_path)
        self.dicom_dir      = dicom_dir
        self.split          = split
        self.label_col      = label_col
        self.num_frames     = num_frames
        self.img_size       = img_size
        self.augment        = augment
        self.strong_augment = strong_augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row        = self.df.iloc[idx]
        patient_id = row['patient_id']
        timepoint  = row['timepoint']

        a4c, has_a4c = _load_video(
            self.dicom_dir, self.split, patient_id, timepoint,
            row['video_a4c'], self.num_frames, self.img_size,
            self.augment, self.strong_augment,
        )
        a2c, has_a2c = _load_video(
            self.dicom_dir, self.split, patient_id, timepoint,
            row['video_a2c'], self.num_frames, self.img_size,
            self.augment, self.strong_augment,
        )

        if not has_a4c and has_a2c:
            a4c     = a2c.clone()
            has_a4c = True

        label        = torch.tensor(float(row[self.label_col]), dtype=torch.float32)
        has_a2c_flag = torch.tensor(float(has_a2c), dtype=torch.float32)

        return a4c, a2c, has_a2c_flag, label
