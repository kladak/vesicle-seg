import numpy as np

from vesicle_seg.eval.metrics import pixel_metrics
from vesicle_seg.models.baseline import otsu_threshold, threshold_baseline
from vesicle_seg.synth.generate import generate_volume


def test_otsu_between_0_255():
    x = np.concatenate([np.full(100, 40), np.full(100, 180)]).astype(np.uint8)
    thr = otsu_threshold(x)
    assert 40 < thr < 180


def test_baseline_returns_binary_same_shape():
    vol = generate_volume(seed=9, shape=(8, 48, 48), n_pools=1)
    pred = threshold_baseline(vol.raw)
    assert pred.shape == vol.raw.shape
    assert set(np.unique(pred)).issubset({0, 1})
    scores = pixel_metrics(pred, vol.labels)
    assert 0.0 <= scores["dice"] <= 1.0
