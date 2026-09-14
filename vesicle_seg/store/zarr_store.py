"""Zarr persistence for synthetic EM-like volumes.

Original writer. Mentions of volara / missionEM_possible in SPEC.md are
problem-shape citations only — no code was ported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import zarr

from vesicle_seg.synth.generate import Volume


def save_volume(store_path: str | Path, volume: Volume, chunks: tuple[int, int, int] | None = None) -> Path:
    store_path = Path(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(store_path), mode="a")
    if volume.volume_id in root:
        del root[volume.volume_id]
    grp = root.create_group(volume.volume_id)
    cks = chunks or _default_chunks(volume.raw.shape)
    grp.create_dataset("raw", data=volume.raw, chunks=cks, overwrite=True)
    grp.create_dataset("labels", data=volume.labels, chunks=cks, overwrite=True)
    grp.create_dataset("instances", data=volume.instances, chunks=cks, overwrite=True)
    grp.attrs["volume_id"] = volume.volume_id
    grp.attrs["voxel_nm"] = [float(v) for v in volume.voxel_nm]
    grp.attrs["n_vesicles"] = int(volume.n_vesicles)
    grp.attrs["seed"] = int(volume.seed)
    grp.attrs["source"] = volume.source
    for key, val in volume.extras.items():
        grp.attrs[str(key)] = val
    return store_path


def save_dataset(store_path: str | Path, volumes: list[Volume]) -> Path:
    store_path = Path(store_path)
    if store_path.exists():
        # Recreate so CI reruns are deterministic.
        import shutil

        shutil.rmtree(store_path)
    for vol in volumes:
        save_volume(store_path, vol)
    return store_path


def load_volume(store_path: str | Path, volume_id: str) -> Volume:
    root = zarr.open_group(str(store_path), mode="r")
    grp = root[volume_id]
    voxel = tuple(float(v) for v in grp.attrs["voxel_nm"])
    extras = {
        k: grp.attrs[k]
        for k in grp.attrs
        if k not in {"volume_id", "voxel_nm", "n_vesicles", "seed", "source"}
    }
    return Volume(
        volume_id=str(grp.attrs["volume_id"]),
        raw=np.asarray(grp["raw"]),
        labels=np.asarray(grp["labels"]),
        instances=np.asarray(grp["instances"]),
        voxel_nm=voxel,  # type: ignore[arg-type]
        n_vesicles=int(grp.attrs["n_vesicles"]),
        seed=int(grp.attrs["seed"]),
        source=str(grp.attrs.get("source", "synthetic")),
        extras=extras,
    )


def list_volume_ids(store_path: str | Path) -> list[str]:
    root = zarr.open_group(str(store_path), mode="r")
    return sorted(root.group_keys())


def volume_tree(store_path: str | Path) -> str:
    root = zarr.open_group(str(store_path), mode="r")
    lines = [f"{store_path}  (zarr group)"]
    for name in sorted(root.group_keys()):
        grp = root[name]
        raw = grp["raw"]
        lines.append(f"  {name}/raw {raw.shape} chunks={raw.chunks} dtype={raw.dtype}")
        lines.append(f"    voxel_nm={list(grp.attrs['voxel_nm'])} n_vesicles={grp.attrs['n_vesicles']}")
    return "\n".join(lines)


def _default_chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(max(1, min(int(s), 32)) for s in shape)


def raw_hash(raw: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()


def store_hashes(store_path: str | Path) -> dict[str, str]:
    ids = list_volume_ids(store_path)
    out: dict[str, str] = {}
    root = zarr.open_group(str(store_path), mode="r")
    for vid in ids:
        out[vid] = raw_hash(np.asarray(root[vid]["raw"]))
    return out


def load_all(store_path: str | Path) -> list[Volume]:
    return [load_volume(store_path, vid) for vid in list_volume_ids(store_path)]
