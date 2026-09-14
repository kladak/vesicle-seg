# Provenance & historical notes

Context for integrity; not required on the front of the README.

## Independent clean-room

This repository implements the *shape* of a Harris Lab (UT Austin) poster pipeline as **educational research tooling** on a public synthetic simulator. It is **not an official Harris Lab release** and is not affiliated as one. No Harris Lab serial-section EM, lab annotations, or internal code are included.

Poster (concepts only): *“Fast vesicle segmentation in neural imaging: An HPC and deep learning approach”* (mentor Vijay Venu).

## Unreconstructible poster figures

A poster draft quotes **~88–92%** accuracy and **“35% better than threshold.”** Those figures are not reconstructible without lab volumes and are **not** written into `reports/metrics.json`. Trust only numbers this repo computes on the simulator.

Vesicle diameters (~40–60 nm) and voxel size `(40, 4, 4)` nm are documented **simulator defaults**, not a dump of lab metadata.

## Related public EM tooling (cited, not copied)

Mentor GitHub: [yajivunev](https://github.com/yajivunev). Problem-shape reference only — **no files from those repos are in this tree**.

- [volara](https://github.com/yajivunev/volara) / [e11bio/volara](https://github.com/e11bio/volara) — block-wise ops on large nD / Zarr volumes
- [missionEM_possible](https://github.com/yajivunev/missionEM_possible) — EM train / predict / IoU eval *layout*
- [EMISAC](https://github.com/yajivunev/EMISAC) — EM stack artifact context

Distinct from [MotionCode](https://github.com/kladak/motioncode) (1-D ECG-like classification).
