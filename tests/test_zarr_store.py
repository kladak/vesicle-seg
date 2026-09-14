from vesicle_seg.store.zarr_store import load_volume, save_volume, volume_tree
from vesicle_seg.synth.generate import generate_volume


def test_zarr_roundtrip(tmp_path):
    vol = generate_volume(seed=4, shape=(8, 32, 32), n_pools=1, vesicles_per_pool=(3, 4))
    store = tmp_path / "toy.zarr"
    save_volume(store, vol)
    back = load_volume(store, vol.volume_id)
    assert (back.raw == vol.raw).all()
    assert (back.labels == vol.labels).all()
    assert back.n_vesicles == vol.n_vesicles
    tree = volume_tree(store)
    assert vol.volume_id in tree
    assert "voxel_nm" in tree
