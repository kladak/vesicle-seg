"""Dice, IoU, F1, pixel accuracy, vesicle density."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import label, uniform_filter


def pixel_metrics(pred: np.ndarray, truth: np.ndarray, eps: float = 1e-8) -> dict[str, float]:
    p = np.asarray(pred).astype(bool)
    t = np.asarray(truth).astype(bool)
    tp = float(np.logical_and(p, t).sum())
    fp = float(np.logical_and(p, ~t).sum())
    fn = float(np.logical_and(~p, t).sum())
    tn = float(np.logical_and(~p, ~t).sum())
    if tp + fp + fn == 0.0:
        dice = iou = prec = rec = f1 = 1.0
    else:
        dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        iou = (tp + eps) / (tp + fp + fn + eps)
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = (2.0 * prec * rec) / (prec + rec + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "pixel_acc": float(acc),
    }


def count_instances(mask: np.ndarray, min_voxels: int = 3) -> int:
    labeled, n = label(np.asarray(mask).astype(bool))
    keep = 0
    for i in range(1, n + 1):
        if int((labeled == i).sum()) >= min_voxels:
            keep += 1
    return keep


def density_summary(
    pred: np.ndarray,
    truth: np.ndarray,
    voxel_nm: tuple[float, float, float],
    n_gt_instances: int | None = None,
) -> dict[str, float]:
    vz, vy, vx = voxel_nm
    um3 = (pred.shape[0] * vz * pred.shape[1] * vy * pred.shape[2] * vx) / 1e9
    um3 = max(um3, 1e-12)
    n_pred = count_instances(pred)
    n_truth = int(n_gt_instances) if n_gt_instances is not None else count_instances(truth)
    return {
        "n_pred": float(n_pred),
        "n_truth": float(n_truth),
        "density_per_um3": float(n_pred / um3),
        "gt_density_per_um3": float(n_truth / um3),
        "density_abs_err": float(abs(n_pred - n_truth) / um3),
        "volume_um3": float(um3),
    }


def density_map(mask: np.ndarray, window: tuple[int, int, int] = (3, 16, 16)) -> np.ndarray:
    """Local vesicle-occupancy heatmap (fraction of window labeled)."""
    return uniform_filter(np.asarray(mask).astype(np.float32), size=window)


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def merge_model_row(pixel: dict[str, float], dens: dict[str, float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out.update(pixel)
    out.update(dens)
    return out
