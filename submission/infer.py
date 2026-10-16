"""EchoRisk-MICCAI 2026 submission inference — EchoPrime backbone.

Adapted from the official reference (submission-example/infer.py). The
harness contract (manifest loading, missing-view convention, row-coverage
guarantee, output schema, per-case fallback) is preserved verbatim; only
`predict` and model setup are replaced with our real EchoNet model.

Contract: docs/HARNESS_SPEC.md.
  - Input  : /input/manifest.csv + /input/videos/{patient}/{timepoint}/{file}
  - Output : /output/predictions.csv with cols [patient_id, timepoint, <task_col>]
  - TASK_ID: baked in at build time via ENV (no run-time args).
  - Every manifest row MUST get a prediction row (fallback on failure).
"""
from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Optional

import numpy as np
import pydicom
import torch
import cv2

from model import EchoNet, ECHOPRIME_ENCODER_WEIGHTS

INPUT_DIR: Final[Path] = Path("/input")
OUTPUT_DIR: Final[Path] = Path("/output")
MANIFEST_PATH: Final[Path] = INPUT_DIR / "manifest.csv"
VIDEOS_DIR: Final[Path] = INPUT_DIR / "videos"
OUTPUT_PATH: Final[Path] = OUTPUT_DIR / "predictions.csv"

REQUIRED_MANIFEST_COLS: Final[tuple[str, ...]] = (
    "patient_id",
    "timepoint",
    "video_a4c",
    "video_a2c",
)

# EchoPrime normalisation stats (raw pixel scale 0-255), identical to training.
_MEAN = torch.tensor([29.110628, 28.076836, 29.096405])
_STD = torch.tensor([47.989223, 46.456997, 47.20083])

NUM_FRAMES: Final[int] = 16
IMG_SIZE: Final[int] = 224


@dataclass(frozen=True)
class TaskConfig:
    output_col: str
    value_min: float
    value_max: float
    fallback: float
    task_type: str          # 'regression' | 'classification'
    checkpoint: str         # path inside the image


_TASK_CONFIGS: Final[dict[int, TaskConfig]] = {
    1: TaskConfig("lvef_pred", 0.0, 100.0, 55.0, "regression", "/app/checkpoints/task1/best.pth"),
    2: TaskConfig("dysfunction_pred", 0.0, 1.0, 0.5, "classification", "/app/checkpoints/task2/best.pth"),
    3: TaskConfig("cardiotoxicity_pred", 0.0, 1.0, 0.5, "classification", "/app/checkpoints/task3/best.pth"),
}


# --------------------------------------------------------------------------
# Frame processing — inlined from src_echoprime/dataset.py (inference path
# only; no augmentation) so the container needs no training-time code.
# --------------------------------------------------------------------------
def _read_dicom_gray(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array
    if arr.ndim == 4:              # (T, H, W, 3) YBR/RGB -> Y/first channel
        gray = arr[..., 0].astype(np.float32)
    elif arr.ndim == 3:           # (T, H, W)
        gray = arr.astype(np.float32)
    elif arr.ndim == 2:           # single frame (H, W) -> (1, H, W)
        gray = arr.astype(np.float32)[None, ...]
    else:
        raise ValueError(f"unexpected DICOM pixel_array ndim={arr.ndim}")
    return gray


def _sample_frames(arr: np.ndarray, n: int) -> np.ndarray:
    T = arr.shape[0]
    indices = np.linspace(0, T - 1, n).round().astype(int)
    indices = np.clip(indices, 0, T - 1)
    return arr[indices]


def _resize_frames(frames: np.ndarray, size: int) -> np.ndarray:
    if frames.shape[1] == size and frames.shape[2] == size:
        return frames
    return np.stack([
        cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR)
        for f in frames
    ])


def _to_tensor(frames: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(frames))    # (T, H, W)
    t = t.unsqueeze(0).expand(3, -1, -1, -1).float()      # (3, T, H, W)
    mean = _MEAN.view(3, 1, 1, 1)
    std = _STD.view(3, 1, 1, 1)
    return (t - mean) / std


def _load_view(path: Optional[Path]) -> tuple[torch.Tensor, bool]:
    """Load one view -> (3, T, H, W) tensor and availability flag.

    Returns a zero tensor + False when the view is absent (path is None).
    Raises on a genuinely corrupt/unreadable file so the caller falls back.
    """
    if path is None:
        return torch.zeros(3, NUM_FRAMES, IMG_SIZE, IMG_SIZE), False
    gray = _read_dicom_gray(path)
    frames = _sample_frames(gray, NUM_FRAMES)
    frames = _resize_frames(frames, IMG_SIZE)
    return _to_tensor(frames), True


# --------------------------------------------------------------------------
# Harness plumbing — preserved from the reference implementation.
# --------------------------------------------------------------------------
def _load_task_id() -> int:
    raw = os.environ.get("TASK_ID")
    if raw is None:
        print("error: TASK_ID is not set (bake in at build time via "
              "--build-arg TASK_ID=<1|2|3>).", file=sys.stderr)
        sys.exit(1)
    try:
        task_id = int(raw)
    except ValueError:
        print(f"error: TASK_ID must be an integer, got '{raw}'.", file=sys.stderr)
        sys.exit(1)
    if task_id not in _TASK_CONFIGS:
        print(f"error: TASK_ID must be 1, 2, or 3; got {task_id}.", file=sys.stderr)
        sys.exit(1)
    return task_id


def _load_manifest() -> list[dict[str, str]]:
    if not MANIFEST_PATH.exists():
        print(f"error: manifest not found at {MANIFEST_PATH}.", file=sys.stderr)
        sys.exit(1)
    with MANIFEST_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("error: manifest.csv has no header row.", file=sys.stderr)
            sys.exit(1)
        missing = set(REQUIRED_MANIFEST_COLS) - set(reader.fieldnames)
        if missing:
            print(f"error: manifest missing required columns: {sorted(missing)}.",
                  file=sys.stderr)
            sys.exit(1)
        rows = list(reader)
    if not rows:
        print("error: manifest.csv contains no case rows.", file=sys.stderr)
        sys.exit(1)
    return rows


def _resolve_view(filename: str) -> Optional[Path]:
    """Resolve a manifest video cell to a path.

    Missing view is signalled by an EMPTY manifest cell (contract §2.4),
    not by file absence. A non-empty cell pointing at a nonexistent file is
    a data-integrity error -> raise so the row falls back (contract §4).
    The manifest value is a full relative path under /input/videos/.
    """
    if filename.strip() == "":
        return None
    path = VIDEOS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"referenced video not found: {path}")
    return path


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def _build_model(config: TaskConfig, device: torch.device) -> EchoNet:
    # Task 3 was trained with freeze_backbone=True; the flag only affects
    # requires_grad, not the forward pass, but we match training for parity.
    freeze = (config.task_type == "classification" and
              config.output_col == "cardiotoxicity_pred")
    model = EchoNet(
        task=config.task_type,
        freeze_backbone=freeze,
        encoder_weights=ECHOPRIME_ENCODER_WEIGHTS,
    ).to(device)
    ckpt = torch.load(config.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def predict(
    model: EchoNet,
    config: TaskConfig,
    device: torch.device,
    a4c_path: Optional[Path],
    a2c_path: Optional[Path],
) -> float:
    """Run inference on one case. Raising is safe — caller falls back (§4)."""
    a4c, has_a4c = _load_view(a4c_path)
    a2c, has_a2c = _load_view(a2c_path)

    # Match training: if only A2C present, mirror it into the A4C slot.
    if not has_a4c and has_a2c:
        a4c, has_a4c = a2c.clone(), True

    if not has_a4c and not has_a2c:
        # Both views absent -> no signal. Fall back (handled by caller).
        raise ValueError("both views absent")

    a4c_b = a4c.unsqueeze(0).to(device)
    a2c_b = a2c.unsqueeze(0).to(device)
    has_a2c_b = torch.tensor([float(has_a2c)], device=device)

    use_amp = device.type == "cuda"
    with torch.autocast(device_type=device.type, enabled=use_amp):
        logit = model(a4c_b, a2c_b, has_a2c_b)

    if config.task_type == "regression":
        return float(logit.float().clamp(config.value_min, config.value_max).item())
    return float(torch.sigmoid(logit.float()).item())


def main() -> None:
    task_id = _load_task_id()
    config = _TASK_CONFIGS[task_id]
    manifest_rows = _load_manifest()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"TASK_ID={task_id} | device={device} | cases={len(manifest_rows)}")

    # Model loaded ONCE (contract §7 wall-clock budget includes load time).
    model = _build_model(config, device)
    print(f"Loaded model from {config.checkpoint}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_fallback = 0

    with OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "timepoint", config.output_col])
        writer.writeheader()
        for row in manifest_rows:
            patient_id = row["patient_id"]
            timepoint = row["timepoint"]
            try:
                a4c_path = _resolve_view(row["video_a4c"])
                a2c_path = _resolve_view(row["video_a2c"])
                value = predict(model, config, device, a4c_path, a2c_path)
                if not (config.value_min <= value <= config.value_max):
                    raise ValueError(f"prediction {value} out of range")
            except Exception as exc:  # noqa: BLE001 — row-boundary catch-all (§4)
                print(f"warning: case ({patient_id}, {timepoint}) failed "
                      f"({exc!r}); using fallback {config.fallback}.", file=sys.stderr)
                value = config.fallback
                n_fallback += 1
            writer.writerow({"patient_id": patient_id, "timepoint": timepoint,
                             config.output_col: value})

    print(f"Wrote {len(manifest_rows)} predictions to {OUTPUT_PATH} "
          f"({n_fallback} used fallback).")


if __name__ == "__main__":
    main()
