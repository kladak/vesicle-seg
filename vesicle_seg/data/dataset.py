"""Chunk dataset drawn from whole volumes (volume_id stays with every chunk)."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from vesicle_seg.chunking.tiles import plan_chunks, read_chunk
from vesicle_seg.synth.generate import Volume


class VolumeChunkDataset(Dataset):
    def __init__(
        self,
        volumes: list[Volume],
        chunk: tuple[int, int, int],
        overlap: tuple[int, int, int] | int,
        min_edge_frac: float = 0.35,
    ) -> None:
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        self.volume_ids: list[str] = []
        for vol in volumes:
            for spec in plan_chunks(vol.shape, chunk, overlap, min_edge_frac):
                raw = read_chunk(vol.raw, spec)
                lab = read_chunk(vol.labels, spec)
                raw_t, lab_t = _pad_to(raw, lab, chunk)
                self.samples.append((raw_t, lab_t))
                self.volume_ids.append(vol.volume_id)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raw, lab = self.samples[idx]
        x = torch.from_numpy(raw.astype(np.float32) / 255.0).unsqueeze(0)
        x = (x - x.mean()) / (x.std() + 1e-6)
        y = torch.from_numpy(lab.astype(np.float32)).unsqueeze(0)
        return x, y


def _pad_to(
    raw: np.ndarray,
    lab: np.ndarray,
    chunk: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    pads = []
    for dim, size in enumerate(chunk):
        need = size - raw.shape[dim]
        if need < 0:
            raise ValueError("chunk smaller than read region")
        pads.append((0, need))
    if any(p[1] for p in pads):
        raw = np.pad(raw, pads, mode="edge")
        lab = np.pad(lab, pads, mode="constant")
    return raw, lab
