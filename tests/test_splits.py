import pytest

from vesicle_seg.data.splits import grouped_split, leakage_report
from vesicle_seg.synth.generate import generate_dataset


def test_grouped_split_no_id_or_hash_leakage():
    volumes = generate_dataset(n_volumes=6, seed=42, shape=(8, 32, 32), n_pools=1, vesicles_per_pool=(3, 4))
    split = grouped_split(volumes, seed=0, fractions=(0.5, 0.17, 0.33))
    assert split.leakage_ok
    bags = [set(split.train), set(split.val), set(split.test)]
    assert bags[0].isdisjoint(bags[1])
    assert bags[0].isdisjoint(bags[2])
    assert bags[1].isdisjoint(bags[2])
    ok, notes = leakage_report(split)
    assert ok and notes == []


def test_duplicate_id_rejected():
    volumes = generate_dataset(n_volumes=2, seed=1, shape=(8, 32, 32), n_pools=1)
    volumes[1].volume_id = volumes[0].volume_id
    with pytest.raises(ValueError):
        grouped_split(volumes)
