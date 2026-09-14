"""CLI: generate, train, infer, demo, serve."""

from __future__ import annotations

import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from vesicle_seg.config import load_config, shape_tuple, voxel_nm_tuple
from vesicle_seg.infer import load_model, predict_volume
from vesicle_seg.models.baseline import threshold_baseline
from vesicle_seg.store.zarr_store import load_volume, save_dataset, volume_tree
from vesicle_seg.synth.generate import generate_dataset
from vesicle_seg.train import train_from_config
from vesicle_seg.viz import save_comparison, save_panel


def _cmd_generate(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    volumes = generate_dataset(
        n_volumes=int(cfg["n_volumes"]),
        seed=int(cfg["seed"]),
        shape=shape_tuple(cfg),
        voxel_nm=voxel_nm_tuple(cfg),
        n_pools=int(cfg.get("n_pools", 2)),
        vesicles_per_pool=tuple(cfg.get("vesicles_per_pool", [6, 12])),
        n_mito=int(cfg.get("n_mito", 1)),
    )
    path = Path(args.out or cfg.get("zarr_path", "data/synthetic.zarr"))
    save_dataset(path, volumes)
    print(volume_tree(path))


def _cmd_train(args: argparse.Namespace) -> None:
    metrics = train_from_config(args.config)
    print(json.dumps({"leakage_ok": metrics["leakage_ok"], "models": metrics["models"]}, indent=2))
    print(metrics["poster_draft_note"])


def _cmd_infer(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    store = Path(args.store or cfg.get("zarr_path", "data/synthetic.zarr"))
    vol = load_volume(store, args.volume_id)
    reports = Path(cfg.get("reports_dir", "reports"))
    reports.mkdir(parents=True, exist_ok=True)
    base = threshold_baseline(vol.raw)
    weights = Path(args.weights or cfg.get("artifacts_dir", "artifacts") + "/resunet3d.pt")
    if weights.exists():
        device = torch.device("cpu")
        model = load_model(
            weights,
            device,
            base=int(cfg.get("base_channels", 8)),
            dropout=float(cfg.get("dropout", 0.1)),
        )
        pred = predict_volume(
            model,
            vol.raw,
            tuple(cfg["chunk"]),
            tuple(cfg["overlap"]),
            device,
        )
        save_panel(reports / "overlay_panel.png", vol.raw, vol.labels, base, pred)
        save_comparison(reports / "overlay_unet.png", vol.raw, vol.labels, pred, f"infer {vol.volume_id}")
        print(f"wrote {reports / 'overlay_panel.png'}")
    else:
        save_comparison(reports / "overlay_threshold.png", vol.raw, vol.labels, base, f"threshold {vol.volume_id}")
        print(f"no weights at {weights}; wrote threshold overlay only")


def _cmd_demo(args: argparse.Namespace) -> None:
    metrics = train_from_config(args.config)
    print("=== vesicle-seg demo (synthetic only) ===")
    print(metrics["zarr_tree"])
    print()
    print("split", metrics["split"], "leakage_ok=", metrics["leakage_ok"])
    print("threshold", metrics["models"]["threshold"])
    print("resunet3d", metrics["models"]["resunet3d"])
    print()
    print(metrics["poster_draft_note"])
    print("overlays: reports/overlay_panel.png  reports/overlay_unet.png")
    print("metrics:  reports/metrics.json")


def _cmd_serve(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    handler = SimpleHTTPRequestHandler
    # Serve repo root so viewer/index.html can fetch reports/*.png
    import os

    os.chdir(root)
    httpd = ThreadingHTTPServer(("127.0.0.1", int(args.port)), handler)
    print(f"static viewer  http://127.0.0.1:{args.port}/viewer/")
    print("Ctrl-C to stop. Educational overlays only.")
    httpd.serve_forever()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vesicle-seg", description="Synthetic synaptic-vesicle segmentation")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write synthetic Zarr volumes")
    g.add_argument("--config", default="configs/ci.yaml")
    g.add_argument("--out", default=None)
    g.set_defaults(func=_cmd_generate)

    t = sub.add_parser("train", help="train residual 3D U-Net + baseline")
    t.add_argument("--config", default="configs/ci.yaml")
    t.set_defaults(func=_cmd_train)

    i = sub.add_parser("infer", help="stitched overlapping-chunk inference")
    i.add_argument("--config", default="configs/ci.yaml")
    i.add_argument("--volume-id", dest="volume_id", required=True)
    i.add_argument("--store", default=None)
    i.add_argument("--weights", default=None)
    i.set_defaults(func=_cmd_infer)

    d = sub.add_parser("demo", help="generate + train + overlays (60–90s script)")
    d.add_argument("--config", default="configs/ci.yaml")
    d.set_defaults(func=_cmd_demo)

    s = sub.add_parser("serve", help="static overlay viewer")
    s.add_argument("--port", default=8765)
    s.add_argument("--root", default=".")
    s.set_defaults(func=_cmd_serve)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
