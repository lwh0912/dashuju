"""Numerically stable single-logit focal loss; no implicit class weighting."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and nonnegative")
        self.gamma = gamma

    def forward(self, logits, targets):
        if logits.shape != targets.shape:
            raise ValueError("Single-logit predictions and targets must have the same shape")
        ce = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
        # Clamp protects fractional gamma gradients at perfectly saturated logits.
        modulation = (-torch.expm1(-ce)).clamp_min(torch.finfo(ce.dtype).tiny)
        return (modulation.pow(self.gamma) * ce).mean()
