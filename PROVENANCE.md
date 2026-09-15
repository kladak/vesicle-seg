# Provenance

## Problem origin

The pipeline follows the shape of a Harris Lab (UT Austin) poster,
*"Fast vesicle segmentation in neural imaging: An HPC and deep learning approach"*
(mentor: Vijay Venu). Volumes and labels come from the generator in `vesicle_seg/synth/`.

## Simulator parameters

Vesicle diameters (~40-60 nm) and voxel size `(40, 4, 4)` nm are the generator defaults,
chosen to match ssEM-like anisotropy.

## Related public EM tooling

These informed the design and are cited for reference.

Mentor GitHub: [yajivunev](https://github.com/yajivunev).

- [volara](https://github.com/yajivunev/volara) / [e11bio/volara](https://github.com/e11bio/volara):
  block-wise operations on large nD and Zarr volumes
- [missionEM_possible](https://github.com/yajivunev/missionEM_possible): EM train, predict
  and IoU evaluation layout
- [EMISAC](https://github.com/yajivunev/EMISAC): EM stack artifact context
