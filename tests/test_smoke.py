"""End-to-end smoke on a two-volume toy config (still synthetic)."""

from pathlib import Path

from vesicle_seg.train import run_experiment


def test_train_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = {
        "seed": 42,
        "n_volumes": 4,
        "shape": [8, 32, 32],
        "voxel_nm": [40.0, 4.0, 4.0],
        "n_pools": 1,
        "vesicles_per_pool": [4, 6],
        "n_mito": 1,
        "chunk": [8, 32, 32],
        "overlap": [0, 0, 0],
        "min_edge_frac": 0.35,
        "fractions": [0.5, 0.25, 0.25],
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.01,
        "base_channels": 8,
        "dropout": 0.0,
        "lambda_focal": 0.5,
        "zarr_path": str(tmp_path / "toy.zarr"),
        "reports_dir": str(tmp_path / "reports"),
        "artifacts_dir": str(tmp_path / "artifacts"),
    }
    metrics = run_experiment(cfg, tmp_path)
    assert metrics["source"] == "synthetic"
    assert metrics["leakage_ok"] is True
    assert "resunet3d" in metrics["models"]
    assert "threshold" in metrics["models"]
    assert 0.0 <= metrics["models"]["resunet3d"]["dice"] <= 1.0
    assert Path(tmp_path / "reports" / "metrics.json").is_file()
    assert Path(tmp_path / "reports" / "overlay_panel.png").is_file()
