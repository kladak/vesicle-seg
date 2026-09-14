import numpy as np

from vesicle_seg.chunking.tiles import plan_chunks, read_chunk, stitch_inner


def test_plan_covers_every_voxel():
    shape = (8, 64, 64)
    specs = plan_chunks(shape, chunk=(8, 32, 32), overlap=(0, 4, 4))
    assert specs
    covered = np.zeros(shape, dtype=np.int32)
    for spec in specs:
        covered[spec.write_slices] += 1
    assert (covered >= 1).all()


def test_halo_is_not_written_in_the_interior():
    specs = plan_chunks((8, 64, 64), chunk=(8, 32, 32), overlap=(0, 4, 4))
    # A non-border tile should have a write region smaller than the read region in Y/X.
    interiors = [s for s in specs if s.origin[1] not in (0, 32) or s.origin[2] not in (0, 32)]
    if interiors:
        spec = interiors[0]
        write = tuple(s.stop - s.start for s in spec.write_slices)
        read = spec.read_shape
        assert write[1] <= read[1]
        assert write[2] <= read[2]


def test_stitch_roundtrip_inner():
    vol = np.arange(8 * 32 * 32, dtype=np.int32).reshape(8, 32, 32)
    specs = plan_chunks(vol.shape, chunk=(8, 32, 32), overlap=(0, 4, 4))
    out = np.zeros_like(vol)
    for spec in specs:
        stitch_inner(out, spec, read_chunk(vol, spec))
    assert (out == vol).all()
