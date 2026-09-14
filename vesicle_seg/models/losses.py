"""Dice + focal losses (poster-aligned combination)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets = targets.float()
    dims = tuple(range(1, probs.ndim))
    inter = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.75,
) -> torch.Tensor:
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, probs, 1.0 - probs)
    loss = (alpha * (1.0 - pt).pow(gamma) * bce).mean()
    return loss


def combined_dice_focal(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lambda_focal: float = 0.5,
    gamma: float = 2.0,
) -> torch.Tensor:
    return dice_loss(logits, targets) + float(lambda_focal) * focal_loss(
        logits, targets, gamma=gamma
    )
