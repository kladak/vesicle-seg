from vesicle_seg.synth.generate import generate_volume


def test_seed_is_deterministic():
    a = generate_volume(seed=7, shape=(8, 48, 48), n_pools=1, vesicles_per_pool=(4, 5))
    b = generate_volume(seed=7, shape=(8, 48, 48), n_pools=1, vesicles_per_pool=(4, 5))
    assert (a.raw == b.raw).all()
    assert (a.labels == b.labels).all()
    assert (a.instances == b.instances).all()


def test_different_seeds_differ():
    a = generate_volume(seed=1, shape=(8, 48, 48))
    b = generate_volume(seed=2, shape=(8, 48, 48))
    assert not (a.raw == b.raw).all()


def test_vesicles_have_plausible_footprint():
    vol = generate_volume(seed=3, shape=(8, 64, 64), n_pools=2)
    assert vol.n_vesicles >= 4
    assert vol.labels.max() == 1
    # ~40–60 nm at 4 nm/px → roughly 5–15 px across; instance should not fill the crop.
    assert 20 < int(vol.labels.sum()) < vol.raw.size * 0.25
    assert vol.source == "synthetic"
    assert vol.physical_um3() > 0
