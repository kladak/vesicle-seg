"""Threshold + morphology baseline.

The poster draft's “35% better than threshold” figure is a lab claim and is
not reconstructed here. This baseline exists so the synthetic U-Net has a
floor to beat — or not — on the simulator.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_opening, generate_binary_structure, label


def otsu_threshold(values: np.ndarray) -> float:
    hist, edges = np.histogram(values.ravel(), bins=64, range=(0, 255))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 128.0
    centers = 0.5 * (edges[:-1] + edges[1:])
    w1 = np.cumsum(hist)
    w2 = total - w1
    s1 = np.cumsum(hist * centers)
    s2 = s1[-1] - s1
    valid = (w1 > 0) & (w2 > 0)
    if not np.any(valid):
        return float(np.median(values))
    mean1 = s1[valid] / w1[valid]
    mean2 = s2[valid] / w2[valid]
    var = w1[valid] * w2[valid] * (mean1 - mean2) ** 2
    idx = int(np.argmax(var))
    return float(centers[valid][idx])


def threshold_baseline(
    raw: np.ndarray,
    min_voxels: int = 4,
    max_voxels: int = 400,
    opening: int = 1,
) -> np.ndarray:
    """Dark-object threshold. Vesicles *and* membranes/mito fire — by design."""
    raw = np.asarray(raw)
    thr = otsu_threshold(raw)
    # Vesicles are dark; keep pixels below threshold.
    mask = raw < thr
    if opening > 0:
        st = generate_binary_structure(3, 1)
        mask = binary_opening(mask, structure=st, iterations=int(opening))
    labeled, n = label(mask)
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, n + 1):
        sel = labeled == i
        size = int(sel.sum())
        if min_voxels <= size <= max_voxels:
            out[sel] = 1
    return out
