"""Train residual 3-D U-Net + evaluate threshold baseline on held-out volumes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from vesicle_seg.config import load_config, shape_tuple, voxel_nm_tuple
from vesicle_seg.data.dataset import VolumeChunkDataset
from vesicle_seg.data.splits import grouped_split
from vesicle_seg.eval.metrics import density_summary, mean_metrics, merge_model_row, pixel_metrics
from vesicle_seg.infer import predict_volume
from vesicle_seg.models.baseline import threshold_baseline
from vesicle_seg.models.losses import combined_dice_focal
from vesicle_seg.models.residual_unet3d import ResidualUNet3d
from vesicle_seg.store.zarr_store import save_dataset, volume_tree
from vesicle_seg.synth.generate import Volume, generate_dataset
from vesicle_seg.viz import save_comparison, save_panel

DATA_NOTE = "Scores computed on volumes from the in-repo synthetic generator."


def _device() -> torch.device:
    return torch.device("cpu")


def _select(volumes: list[Volume], ids: list[str]) -> list[Volume]:
    index = {v.volume_id: v for v in volumes}
    return [index[i] for i in ids]


def train_model(
    train_vols: list[Volume],
    val_vols: list[Volume],
    cfg: dict[str, Any],
    device: torch.device,
) -> ResidualUNet3d:
    chunk = tuple(int(x) for x in cfg["chunk"])
    overlap = tuple(int(x) for x in cfg["overlap"])
    ds = VolumeChunkDataset(train_vols, chunk, overlap, cfg.get("min_edge_frac", 0.35))  # type: ignore[arg-type]
    loader = DataLoader(ds, batch_size=int(cfg.get("batch_size", 2)), shuffle=True)
    model = ResidualUNet3d(
        base=int(cfg.get("base_channels", 8)),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 1e-3)))
    epochs = int(cfg.get("epochs", 3))
    lam = float(cfg.get("lambda_focal", 0.5))
    model.train()
    for _epoch in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = combined_dice_focal(logits, yb, lambda_focal=lam)
            loss.backward()
            opt.step()
    model.eval()
    # val is used only as a hold-out identity in v0 (no early stopping).
    _ = val_vols
    return model


def evaluate_model(
    volumes: list[Volume],
    predict_fn,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    rows = []
    for vol in volumes:
        pred = predict_fn(vol)
        pix = pixel_metrics(pred, vol.labels)
        dens = density_summary(pred, vol.labels, vol.voxel_nm, n_gt_instances=vol.n_vesicles)
        rows.append(merge_model_row(pix, dens))
    return mean_metrics(rows), rows


def run_experiment(cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = Path(cfg.get("reports_dir", "reports"))
    reports.mkdir(parents=True, exist_ok=True)
    artifacts = Path(cfg.get("artifacts_dir", "artifacts"))
    artifacts.mkdir(parents=True, exist_ok=True)

    shape = shape_tuple(cfg)
    voxel = voxel_nm_tuple(cfg)
    volumes = generate_dataset(
        n_volumes=int(cfg["n_volumes"]),
        seed=int(cfg["seed"]),
        shape=shape,
        voxel_nm=voxel,
        n_pools=int(cfg.get("n_pools", 2)),
        vesicles_per_pool=tuple(cfg.get("vesicles_per_pool", [6, 12])),
        n_mito=int(cfg.get("n_mito", 1)),
    )
    store = Path(cfg.get("zarr_path", "data/synthetic.zarr"))
    save_dataset(store, volumes)
    split = grouped_split(
        volumes,
        seed=int(cfg["seed"]),
        fractions=tuple(cfg.get("fractions", [0.7, 0.15, 0.15])),
    )
    train_vols = _select(volumes, split.train)
    val_vols = _select(volumes, split.val) or train_vols[:1]
    test_vols = _select(volumes, split.test) or val_vols

    device = _device()
    model = train_model(train_vols, val_vols, cfg, device)
    weights = artifacts / "resunet3d.pt"
    torch.save(model.state_dict(), weights)

    chunk = tuple(int(x) for x in cfg["chunk"])
    overlap = tuple(int(x) for x in cfg["overlap"])

    def unet_pred(vol: Volume) -> np.ndarray:
        return predict_volume(model, vol.raw, chunk, overlap, device)  # type: ignore[arg-type]

    def base_pred(vol: Volume) -> np.ndarray:
        return threshold_baseline(vol.raw)

    unet_mean, _ = evaluate_model(test_vols, unet_pred)
    base_mean, _ = evaluate_model(test_vols, base_pred)

    demo = test_vols[0]
    base_mask = base_pred(demo)
    unet_mask = unet_pred(demo)
    save_panel(reports / "overlay_panel.png", demo.raw, demo.labels, base_mask, unet_mask)
    save_comparison(
        reports / "overlay_unet.png",
        demo.raw,
        demo.labels,
        unet_mask,
        title=f"resunet3d on {demo.volume_id} (synthetic)",
    )
    save_comparison(
        reports / "overlay_threshold.png",
        demo.raw,
        demo.labels,
        base_mask,
        title=f"threshold on {demo.volume_id} (synthetic)",
    )

    metrics = {
        "source": "synthetic",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(cfg["seed"]),
        "n_volumes": int(cfg["n_volumes"]),
        "shape": list(shape),
        "voxel_nm": list(voxel),
        "chunk": list(chunk),
        "overlap": list(overlap),
        "epochs": int(cfg.get("epochs", 3)),
        "zarr": str(store),
        "zarr_tree": volume_tree(store),
        "split": {"train": split.train, "val": split.val, "test": split.test},
        "leakage_ok": bool(split.leakage_ok),
        "data_note": DATA_NOTE,
        "models": {
            "threshold": {k: round(v, 6) for k, v in base_mean.items()},
            "resunet3d": {k: round(v, 6) for k, v in unet_mean.items()},
        },
        "demo_volume_id": demo.volume_id,
        "weights": str(weights),
    }
    dest = reports / "metrics.json"
    dest.write_text(json.dumps(metrics, indent=2) + "\n")
    example = reports / "metrics.example.json"
    example.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def train_from_config(config_path: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    return run_experiment(cfg, Path("."))
