"""YAML experiment config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} must be a mapping")
    return cfg


def shape_tuple(cfg: dict[str, Any], key: str = "shape") -> tuple[int, int, int]:
    value = cfg[key]
    z, y, x = (int(v) for v in value)
    return z, y, x


def voxel_nm_tuple(cfg: dict[str, Any]) -> tuple[float, float, float]:
    value = cfg.get("voxel_nm", [40.0, 4.0, 4.0])
    z, y, x = (float(v) for v in value)
    return z, y, x
