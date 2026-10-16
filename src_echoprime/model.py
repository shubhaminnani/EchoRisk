"""
EchoNet: EchoPrime MViT-v2-S backbone with dual-view (A4C + A2C) fusion.

Architecture:
  - MViT-v2-S backbone pretrained on echocardiography (EchoPrime weights).
  - Head replaced with Linear(768, 512) to produce 512-d per-view embeddings.
  - A2C features are gated to zero when the view is absent (has_a2c == 0).
  - Concatenated 1024-d vector → MLP head → single scalar output.

Input tensors: (B, 3, T, H, W), T=16, H=W=224, normalised to EchoPrime stats.
"""

import torch
import torch.nn as nn
import torchvision

ECHOPRIME_ENCODER_WEIGHTS = (
    '/mnt/e/challenge/echo_risk/archive/EchoPrime/model_data/weights/echo_prime_encoder.pt'
)


def _build_mvit_encoder(weights_path: str, device: torch.device) -> nn.Module:
    """Load MViT-v2-S with EchoPrime head (512-d output)."""
    encoder = torchvision.models.video.mvit_v2_s()
    encoder.head[-1] = nn.Linear(encoder.head[-1].in_features, 512)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    encoder.load_state_dict(state)
    return encoder


class EchoNet(nn.Module):
    def __init__(self, task: str = 'regression', freeze_backbone: bool = False,
                 encoder_weights: str = ECHOPRIME_ENCODER_WEIGHTS):
        """
        Args:
            task:             'regression' or 'classification'
            freeze_backbone:  Freeze backbone weights (used for Task 3 transfer)
            encoder_weights:  Path to EchoPrime encoder .pt file
        """
        super().__init__()
        assert task in ('regression', 'classification')
        self.task = task

        device = torch.device('cpu')  # moved to correct device by caller
        self.backbone = _build_mvit_encoder(encoder_weights, device)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # 2 × 512 concatenated features
        self.head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, T, H, W) → (B, 512)"""
        return self.backbone(x)

    def forward(self, a4c: torch.Tensor, a2c: torch.Tensor,
                has_a2c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a4c:     (B, 3, T, H, W)
            a2c:     (B, 3, T, H, W) — may be all-zeros when view absent
            has_a2c: (B,) float — 1.0 if A2C present, 0.0 otherwise

        Returns:
            (B,) scalar — raw LVEF estimate OR raw logit for classification
        """
        f4 = self._encode(a4c)
        f2 = self._encode(a2c) * has_a2c.unsqueeze(1)
        return self.head(torch.cat([f4, f2], dim=1)).squeeze(1)
