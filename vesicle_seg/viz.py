"""Slice overlays and density-map figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vesicle_seg.eval.metrics import density_map


def pick_z(guide: np.ndarray | None, depth: int) -> int:
    if guide is None:
        return depth // 2
    counts = np.asarray(guide).reshape(guide.shape[0], -1).sum(axis=1)
    if float(counts.max()) <= 0:
        return depth // 2
    return int(np.argmax(counts))


def overlay_rgb(raw2d: np.ndarray, mask2d: np.ndarray, color: tuple[int, int, int] = (255, 64, 64)) -> np.ndarray:
    base = np.stack([raw2d, raw2d, raw2d], axis=-1).astype(np.float32)
    tint = np.zeros_like(base)
    tint[..., 0] = color[0]
    tint[..., 1] = color[1]
    tint[..., 2] = color[2]
    alpha = (mask2d > 0).astype(np.float32) * 0.45
    out = base * (1.0 - alpha[..., None]) + tint * alpha[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def save_comparison(
    path: Path,
    raw: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
    title: str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    z = pick_z(truth, raw.shape[0])
    r, t, p = raw[z], truth[z], pred[z]
    fig, axes = plt.subplots(1, 4, figsize=(12.2, 3.4))
    axes[0].imshow(r, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"raw  (z={z})")
    axes[1].imshow(t, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("GT vesicles")
    axes[2].imshow(overlay_rgb(r, p))
    axes[2].set_title("pred overlay")
    axes[3].imshow(density_map(pred)[z], cmap="magma")
    axes[3].set_title("density map")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def save_panel(
    path: Path,
    raw: np.ndarray,
    truth: np.ndarray,
    baseline: np.ndarray,
    unet: np.ndarray,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    z = pick_z(truth, raw.shape[0])
    r = raw[z]
    fig, axes = plt.subplots(1, 4, figsize=(12.2, 3.4))
    axes[0].imshow(r, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("synthetic EM")
    axes[1].imshow(overlay_rgb(r, truth[z], (60, 220, 90)))
    axes[1].set_title("ground truth")
    axes[2].imshow(overlay_rgb(r, baseline[z], (255, 180, 40)))
    axes[2].set_title("threshold")
    axes[3].imshow(overlay_rgb(r, unet[z], (80, 160, 255)))
    axes[3].set_title("residual 3D U-Net")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Synthetic volumes",
        fontsize=9,
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
