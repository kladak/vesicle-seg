import numpy as np

from pytest import approx

from vesicle_seg.eval.metrics import count_instances, pixel_metrics


def test_perfect_mask():
    m = np.zeros((4, 8, 8), dtype=np.uint8)
    m[1:3, 2:5, 2:5] = 1
    s = pixel_metrics(m, m)
    assert s["dice"] == approx(1.0)
    assert s["iou"] == approx(1.0)
    assert s["f1"] == approx(1.0)


def test_empty_vs_empty():
    z = np.zeros((2, 4, 4), dtype=np.uint8)
    s = pixel_metrics(z, z)
    assert s["dice"] == 1.0 or s["pixel_acc"] == 1.0


def test_instance_count():
    m = np.zeros((4, 16, 16), dtype=np.uint8)
    m[1, 2:5, 2:5] = 1
    m[2, 10:13, 10:13] = 1
    assert count_instances(m) == 2
