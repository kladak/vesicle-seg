"""Deterministic synthetic ssEM-like volumes with ~40–60 nm vesicle blobs.

Builds ssEM-like volumes from documented parameters; see PROVENANCE.md.
Physical defaults are documented in SPEC.md (coarse ssEM-like anisotropy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class Volume:
    volume_id: str
    raw: np.ndarray  # uint8 (D, H, W)
    labels: np.ndarray  # uint8 (D, H, W)  1 = vesicle
    instances: np.ndarray  # uint16 (D, H, W)
    voxel_nm: tuple[float, float, float]
    n_vesicles: int
    seed: int
    source: str = "synthetic"
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.raw.shape)  # type: ignore[return-value]

    def physical_um3(self) -> float:
        z, y, x = self.shape
        vz, vy, vx = self.voxel_nm
        return (z * vz * y * vy * x * vx) / 1e9


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _cytoplasm(rng: np.random.Generator, shape: tuple[int, int, int]) -> np.ndarray:
    coarse = gaussian_filter(rng.normal(0.0, 1.0, size=shape), sigma=(0.6, 3.5, 3.5))
    fine = gaussian_filter(rng.normal(0.0, 1.0, size=shape), sigma=(0.2, 0.8, 0.8))
    field = 0.70 * _unit(coarse) + 0.30 * _unit(fine)
    return 118.0 + 28.0 * field


def _unit(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _membranes(rng: np.random.Generator, canvas: np.ndarray) -> None:
    """Dark curved ridges: threshold bait, unlabeled."""
    d, h, w = canvas.shape
    n_ridges = max(2, int(round(h * w / 1800)))
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(n_ridges):
        z0 = int(rng.integers(0, d))
        amp = float(rng.uniform(4.0, 12.0))
        freq = float(rng.uniform(0.04, 0.12))
        phase = float(rng.uniform(0, 2 * np.pi))
        thickness = float(rng.uniform(0.8, 1.8))
        axis = int(rng.integers(0, 2))
        if axis == 0:
            curve = (h / 2) + amp * np.sin(freq * xx[0] * 2 * np.pi + phase)
            dist = np.abs(yy - curve[None, :])
        else:
            curve = (w / 2) + amp * np.sin(freq * yy[:, 0] * 2 * np.pi + phase)
            dist = np.abs(xx - curve[:, None])
        ridge = np.exp(-0.5 * (dist / thickness) ** 2)
        z_span = range(max(0, z0 - 1), min(d, z0 + 2))
        for z in z_span:
            canvas[z] -= 38.0 * ridge


def _ellipsoid_mask(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
) -> np.ndarray:
    zz, yy, xx = np.ogrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    cz, cy, cx = center
    rz, ry, rx = radii
    # Avoid zero radii (vesicles are often 1 section thick).
    rz = max(rz, 0.55)
    ry = max(ry, 0.8)
    rx = max(rx, 0.8)
    return ((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0


def _paint_vesicle(
    canvas: np.ndarray,
    labels: np.ndarray,
    instances: np.ndarray,
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
    instance_id: int,
) -> bool:
    mask = _ellipsoid_mask(canvas.shape, center, radii)
    if int(mask.sum()) < 4:
        return False
    # Dark rim with a slightly paler lumen, the simulator's vesicle cue.
    inner_r = tuple(0.62 * r for r in radii)
    inner = _ellipsoid_mask(canvas.shape, center, inner_r)
    rim = mask & ~inner
    canvas[rim] = np.clip(canvas[rim] * 0.42 + 18.0, 0, 255)
    canvas[inner] = np.clip(canvas[inner] * 0.78 + 36.0, 0, 255)
    labels[mask] = 1
    instances[mask] = instance_id
    return True


def _paint_mito(
    canvas: np.ndarray,
    rng: np.random.Generator,
    shape: tuple[int, int, int],
) -> None:
    """Unlabeled elongated dark oval: why a global threshold fails."""
    d, h, w = shape
    center = (
        float(rng.uniform(1, max(1.1, d - 1))),
        float(rng.uniform(8, max(9, h - 8))),
        float(rng.uniform(8, max(9, w - 8))),
    )
    radii = (
        float(rng.uniform(1.2, min(2.8, d * 0.45))),
        float(rng.uniform(6.0, 11.0)),
        float(rng.uniform(3.5, 6.5)),
    )
    mask = _ellipsoid_mask(shape, center, radii)
    canvas[mask] = np.clip(canvas[mask] * 0.28 + 12.0, 0, 255)
    # Faint internal striations (cristae-like, not a biology claim).
    yy = np.arange(h, dtype=np.float32)[None, :, None]
    stripes = 0.5 + 0.5 * np.sin(yy / 1.6)
    canvas[mask] = np.clip(canvas[mask] * (0.85 + 0.2 * np.broadcast_to(stripes, canvas.shape)[mask]), 0, 255)


def generate_volume(
    seed: int,
    shape: tuple[int, int, int] = (8, 64, 64),
    voxel_nm: tuple[float, float, float] = (40.0, 4.0, 4.0),
    n_pools: int = 2,
    vesicles_per_pool: tuple[int, int] = (6, 14),
    vesicle_diameter_nm: tuple[float, float, float] = (40.0, 60.0),
    n_mito: int = 1,
    volume_id: str | None = None,
) -> Volume:
    """Build one synthetic volume. Same seed → bitwise-identical arrays."""
    rng = _rng(int(seed))
    d, h, w = (int(s) for s in shape)
    shape = (d, h, w)
    canvas = _cytoplasm(rng, shape)
    _membranes(rng, canvas)

    for _ in range(int(n_mito)):
        _paint_mito(canvas, rng, shape)

    labels = np.zeros(shape, dtype=np.uint8)
    instances = np.zeros(shape, dtype=np.uint16)
    vz, vy, vx = voxel_nm
    dmin, dmax = vesicle_diameter_nm
    instance_id = 0

    for _ in range(int(n_pools)):
        # "Active-zone" cluster center, kept off the border so vesicles fit.
        pool = (
            float(rng.uniform(1.0, max(1.2, d - 1.2))),
            float(rng.uniform(10.0, max(11.0, h - 10.0))),
            float(rng.uniform(10.0, max(11.0, w - 10.0))),
        )
        n_ves = int(rng.integers(vesicles_per_pool[0], vesicles_per_pool[1] + 1))
        for _k in range(n_ves):
            diam = float(rng.uniform(dmin, dmax))
            radii = (0.5 * diam / vz, 0.5 * diam / vy, 0.5 * diam / vx)
            jitter = (
                float(rng.normal(0.0, 0.7)),
                float(rng.normal(0.0, 4.5)),
                float(rng.normal(0.0, 4.5)),
            )
            center = (
                float(np.clip(pool[0] + jitter[0], 0.4, d - 1.0)),
                float(np.clip(pool[1] + jitter[1], 2.0, h - 3.0)),
                float(np.clip(pool[2] + jitter[2], 2.0, w - 3.0)),
            )
            instance_id += 1
            ok = _paint_vesicle(canvas, labels, instances, center, radii, instance_id)
            if not ok:
                instance_id -= 1

    noise = rng.normal(0.0, 6.5, size=shape)
    canvas = np.clip(canvas + noise, 0.0, 255.0)
    raw = np.rint(canvas).astype(np.uint8)
    n_vesicles = int(instances.max()) if instances.max() > 0 else 0
    vid = volume_id or f"syn_{seed:04d}"
    return Volume(
        volume_id=vid,
        raw=raw,
        labels=labels,
        instances=instances,
        voxel_nm=(float(vz), float(vy), float(vx)),
        n_vesicles=n_vesicles,
        seed=int(seed),
        extras={"n_pools": int(n_pools), "n_mito": int(n_mito)},
    )


def generate_dataset(
    n_volumes: int,
    seed: int,
    shape: tuple[int, int, int] = (8, 64, 64),
    **kwargs: Any,
) -> list[Volume]:
    volumes = []
    for i in range(int(n_volumes)):
        volumes.append(
            generate_volume(
                seed=int(seed) + i * 17,
                shape=shape,
                volume_id=f"syn_{i:03d}",
                **kwargs,
            )
        )
    return volumes
