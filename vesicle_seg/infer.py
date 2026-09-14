"""Overlapping-chunk inference with inner-region stitch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from vesicle_seg.chunking.tiles import plan_chunks, read_chunk, stitch_inner
from vesicle_seg.models.residual_unet3d import ResidualUNet3d


def load_model(path: str | Path, device: torch.device, base: int = 8, dropout: float = 0.1) -> ResidualUNet3d:
    model = ResidualUNet3d(base=base, dropout=dropout)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_volume(
    model: ResidualUNet3d,
    raw: np.ndarray,
    chunk: tuple[int, int, int],
    overlap: tuple[int, int, int] | int,
    device: torch.device,
    min_edge_frac: float = 0.35,
    threshold: float = 0.5,
) -> np.ndarray:
    specs = plan_chunks(raw.shape, chunk, overlap, min_edge_frac)
    logits = np.zeros(raw.shape, dtype=np.float32)
    for spec in specs:
        tile = read_chunk(raw, spec).astype(np.float32) / 255.0
        # Pad if the read region is smaller than the model chunk (volume edge).
        pads = [(0, c - s) for s, c in zip(tile.shape, chunk, strict=True)]
        if any(p[1] for p in pads):
            tile = np.pad(tile, pads, mode="edge")
        tensor = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0)
        tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-6)
        pred = model(tensor.to(device)).squeeze(0).squeeze(0).cpu().numpy()
        pred = pred[: spec.read_shape[0], : spec.read_shape[1], : spec.read_shape[2]]
        stitch_inner(logits, spec, pred)
    return (1.0 / (1.0 + np.exp(-logits)) >= threshold).astype(np.uint8)
