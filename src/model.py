"""
EchoNet: R(2+1)D-18 backbone with dual-view (A4C + A2C) fusion.

Architecture:
  - Shared R(2+1)D-18 backbone (Kinetics-400 pretrained) strips its FC layer.
  - Each view produces a 512-d feature vector after adaptive avg-pool.
  - A2C features are gated to zero when the view is absent (has_a2c == 0).
  - Concatenated 1024-d vector → MLP head → single scalar output.

The same class handles all three tasks via the `task` argument:
  - 'regression'     → raw linear output (clamp to [0,100] at inference)
  - 'classification' → raw logit (apply sigmoid externally for probabilities)
"""

import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


class EchoNet(nn.Module):
    def __init__(self, task: str = 'regression', freeze_backbone: bool = False):
        """
        Args:
            task:             'regression' or 'classification'
            freeze_backbone:  Freeze backbone weights (used for Task 3 transfer)
        """
        super().__init__()
        assert task in ('regression', 'classification')
        self.task = task

        backbone = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        # Drop the FC layer; output after avgpool is (B, 512, 1, 1, 1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # 2 × 512 concatenated features
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, T, H, W) → (B, 512)"""
        # Pretrained backbone expects 3 channels; repeat grayscale channel
        return self.backbone(x.expand(-1, 3, -1, -1, -1)).flatten(1)

    def forward(self, a4c: torch.Tensor, a2c: torch.Tensor,
                has_a2c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a4c:     (B, 1, T, H, W)
            a2c:     (B, 1, T, H, W)  — may be all-zeros when view absent
            has_a2c: (B,) float — 1.0 if A2C present, 0.0 otherwise

        Returns:
            (B,) scalar — raw LVEF estimate OR raw logit for classification
        """
        f4 = self._encode(a4c)
        f2 = self._encode(a2c) * has_a2c.unsqueeze(1)
        return self.head(torch.cat([f4, f2], dim=1)).squeeze(1)
