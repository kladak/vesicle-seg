"""Adaptive overlapping chunks for 3-D volumes.

Halo is read for context and discarded on write so tile seams do not
double-count. If the leftover edge would be a sliver, the last origin is
snapped (and a redundant near-duplicate origin is dropped).

Original planner — not a port of volara / daisy block scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChunkSpec:
    origin: tuple[int, int, int]
    read_slices: tuple[slice, slice, slice]
    write_slices: tuple[slice, slice, slice]
    inner_in_chunk: tuple[slice, slice, slice]

    @property
    def read_shape(self) -> tuple[int, int, int]:
        return tuple(s.stop - s.start for s in self.read_slices)  # type: ignore[return-value]


def _axis_origins(length: int, chunk: int, overlap: int, min_edge_frac: float) -> list[int]:
    """Full-size tiles; last origin snaps to the end so a sliver remainder is absorbed as extra overlap."""
    if length <= chunk:
        return [0]
    halo = max(0, int(overlap))
    inner = max(1, chunk - 2 * halo)
    origins = list(range(0, length - chunk + 1, inner))
    last = length - chunk
    if origins[-1] != last:
        # Adaptive: do not emit a tiny leftover tile — slide a full-size last tile.
        # Keep intermediate origins so inner-write regions still abut (no seams/gaps).
        # Adaptive last tile: always a full-size window snapped to the end
        # (absorbs a sliver remainder as extra overlap, never a thin leftover tile).
        _ = min_edge_frac  # reserved for callers; snapping is the v0 adaptive rule
        origins.append(last)
    # Unique, keep order.
    seen: list[int] = []
    for o in origins:
        if o not in seen:
            seen.append(o)
    return seen


def plan_chunks(
    shape: tuple[int, int, int],
    chunk: tuple[int, int, int],
    overlap: tuple[int, int, int] | int = 2,
    min_edge_frac: float = 0.35,
) -> list[ChunkSpec]:
    """Plan tiles that cover `shape`. Every voxel is in at least one write region."""
    if isinstance(overlap, int):
        overlap = (overlap, overlap, overlap)
    origins_z = _axis_origins(shape[0], chunk[0], overlap[0], min_edge_frac)
    origins_y = _axis_origins(shape[1], chunk[1], overlap[1], min_edge_frac)
    origins_x = _axis_origins(shape[2], chunk[2], overlap[2], min_edge_frac)

    specs: list[ChunkSpec] = []
    for oz in origins_z:
        for oy in origins_y:
            for ox in origins_x:
                origin = (oz, oy, ox)
                read = []
                write = []
                inner = []
                for axis, (o, c, h, length) in enumerate(
                    zip(origin, chunk, overlap, shape, strict=True)
                ):
                    end = min(o + c, length)
                    start = o
                    # If the volume is smaller than the chunk, read the whole axis.
                    if length < c:
                        start, end = 0, length
                    read.append(slice(start, end))
                    # Inner write: drop halo except at volume borders.
                    w0 = start if start == 0 else start + h
                    w1 = end if end == length else end - h
                    if w1 <= w0:
                        w0, w1 = start, end
                    write.append(slice(w0, w1))
                    inner.append(slice(w0 - start, w1 - start))
                specs.append(
                    ChunkSpec(
                        origin=origin,
                        read_slices=tuple(read),  # type: ignore[arg-type]
                        write_slices=tuple(write),  # type: ignore[arg-type]
                        inner_in_chunk=tuple(inner),  # type: ignore[arg-type]
                    )
                )
    _assert_coverage(shape, specs)
    return specs


def _assert_coverage(shape: tuple[int, int, int], specs: list[ChunkSpec]) -> None:
    covered = np.zeros(shape, dtype=np.uint8)
    for spec in specs:
        covered[spec.write_slices] = 1
    if int(covered.sum()) != int(np.prod(shape)):
        missing = int(np.prod(shape) - covered.sum())
        raise RuntimeError(f"chunk plan missed {missing} voxels")


def read_chunk(volume: np.ndarray, spec: ChunkSpec) -> np.ndarray:
    return np.asarray(volume[spec.read_slices])


def stitch_inner(canvas: np.ndarray, spec: ChunkSpec, pred_chunk: np.ndarray) -> None:
    """Write the inner region of `pred_chunk` onto `canvas` (last write wins)."""
    canvas[spec.write_slices] = pred_chunk[spec.inner_in_chunk]
