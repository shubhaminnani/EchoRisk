"""
Shared dataset utilities for all three EchoRisk tasks.

DICOM format: JPEG-compressed, YBR_FULL_422, (T, H, W, 3) uint8.
We extract the Y (luma) channel for grayscale input to the video backbone.
Frames vary from ~25 to 80 per cine loop; we resample to a fixed count.
"""

import os
import random
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset
import cv2


# ImageNet-equivalent stats for grayscale (averaged across RGB channels)
_MEAN = 0.449
_STD  = 0.226


def _read_dicom_gray(path: str) -> np.ndarray:
    """Return (T, H, W) float32 in [0, 1] from a DICOM cine loop."""
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array  # (T, H, W, 3) YBR or (T, H, W) gray
    if arr.ndim == 4:
        # Take Y (luma) channel directly — equivalent to grayscale
        gray = arr[..., 0].astype(np.float32)
    else:
        gray = arr.astype(np.float32)
    return gray / 255.0


def _sample_frames(arr: np.ndarray, n: int, temporal_jitter: bool = False) -> np.ndarray:
    """Uniformly sample n frames from (T, H, W), with optional random start offset."""
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
    """Random horizontal flip + brightness/contrast jitter on (T, H, W)."""
    if random.random() < 0.5:
        frames = frames[:, :, ::-1].copy()
    if random.random() < 0.5:
        alpha = random.uniform(0.8, 1.2)   # contrast
        beta  = random.uniform(-0.08, 0.08) # brightness
        frames = np.clip(alpha * frames + beta, 0.0, 1.0)
    return frames


def _to_tensor(frames: np.ndarray) -> torch.Tensor:
    """(T, H, W) float32 → (1, T, H, W) normalised tensor."""
    t = torch.from_numpy(np.ascontiguousarray(frames))  # (T, H, W)
    return ((t.unsqueeze(0) - _MEAN) / _STD)            # (1, T, H, W)


def _load_video(dicom_dir: str, split: str, patient_id: str,
                timepoint: str, filename, num_frames: int, img_size: int,
                augment: bool) -> tuple[torch.Tensor, bool]:
    """
    Load a single view video. Returns (tensor (1, T, H, W), available: bool).
    Returns a zero tensor when the file is absent or missing.
    """
    if pd.isna(filename) or str(filename).strip() == '':
        return torch.zeros(1, num_frames, img_size, img_size), False

    path = os.path.join(dicom_dir, split, patient_id, timepoint, str(filename))
    if not os.path.exists(path):
        return torch.zeros(1, num_frames, img_size, img_size), False

    gray   = _read_dicom_gray(path)
    frames = _sample_frames(gray, num_frames, temporal_jitter=augment)
    frames = _resize_frames(frames, img_size)
    if augment:
        frames = _augment_spatial(frames)
    return _to_tensor(frames), True


class EchoDataset(Dataset):
    """
    Generic dataset for all three tasks.

    Args:
        csv_path:   Path to the label CSV file (task1_train.csv etc.)
        dicom_dir:  Root that contains train/ and val/ subdirectories.
        split:      'train' or 'val'
        label_col:  Column name for the target label (lvef / lv_dysfunction / cardiotoxicity)
        num_frames: Number of frames to sample per video (default 32)
        img_size:   Spatial resize target in pixels (default 112)
        augment:    Whether to apply data augmentation
    """

    def __init__(self, csv_path: str, dicom_dir: str, split: str,
                 label_col: str, num_frames: int = 32,
                 img_size: int = 112, augment: bool = False):
        self.df         = pd.read_csv(csv_path)
        self.dicom_dir  = dicom_dir
        self.split      = split
        self.label_col  = label_col
        self.num_frames = num_frames
        self.img_size   = img_size
        self.augment    = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row        = self.df.iloc[idx]
        patient_id = row['patient_id']
        timepoint  = row['timepoint']

        a4c, has_a4c = _load_video(
            self.dicom_dir, self.split, patient_id, timepoint,
            row['video_a4c'], self.num_frames, self.img_size, self.augment
        )
        a2c, has_a2c = _load_video(
            self.dicom_dir, self.split, patient_id, timepoint,
            row['video_a2c'], self.num_frames, self.img_size, self.augment
        )

        # When A4C is absent, promote A2C to primary so the backbone always
        # receives a real signal in the first branch
        if not has_a4c and has_a2c:
            a4c    = a2c.clone()
            has_a4c = True

        label = torch.tensor(float(row[self.label_col]), dtype=torch.float32)
        has_a2c_flag = torch.tensor(float(has_a2c), dtype=torch.float32)

        return a4c, a2c, has_a2c_flag, label
