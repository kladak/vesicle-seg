# vesicle-seg — SPEC

**Owner:** Karim Ladak (`kladak`)
**Repo:** https://github.com/kladak/vesicle-seg
**Status:** Greenfield v0 (created 2026-09-14)
**Core question:** *Can we run the Harris Lab poster pipeline — raw EM-like volume → Zarr → adaptive overlapping chunks → residual 3D U-Net (Dice + focal) → density maps — as a reproducible, leakage-aware, offline experiment that is honest about what the numbers mean?*

> **Honesty (read this first).** This is **educational research tooling**, not clinical imaging software and not a medical device. Nothing here is validated on lab or hospital data. Do not use it for patient care, diagnosis, or any biological measurement you would publish as fact.
>
> **Not a Harris Lab data dump.** No serial-section EM volumes, vesicle annotations, or internal lab code from the Kristen Harris laboratory (UT Austin / Center for Learning and Memory) are included or reconstructed. All images and labels are **synthetic**, generated in this repo from documented parameters.
>
> **Not an official lab release.** Independent clean-room build. Mentorship and problem framing come from a public poster and publicly listed mentor tooling; the implementation is original.
>
> **Poster draft figures are not CI metrics.** A circulating poster draft quotes segmentation accuracies around **88–92%** and “**35% better than threshold**.” Those numbers are **not reconstructible** without Harris Lab volumes and are **not** written into `reports/metrics.json`. This repo reports **only** scores computed from the synthetic generator on the current machine.

This project is **distinct from MotionCode** (1-D ECG-like classification). The task here is **3-D semantic segmentation** of synaptic-vesicle-like blobs in EM-like volumes, with block-wise inference and a density summary.

## 1. Problem (from the poster, concepts only)

Poster: **“Fast vesicle segmentation in neural imaging: An HPC and deep learning approach”** (Harris Lab, UT Austin; mentor Vijay Venu).

The scientific object is the **synaptic vesicle** — a ~40–60 nm sphere packed at active zones in neuropil, imaged with serial-section electron microscopy. Counting those vesicles (and mapping their local density) is a bottleneck: volumes are large, membranes and mitochondria look similar under a global threshold, and naive tiles leave seams at chunk boundaries.

Poster pipeline this repo re-implements as a **teaching skeleton**:

```text
Raw EM-like volume
  → Zarr (chunked, attribute-rich store)
  → Adaptive chunking with overlap (halo discarded / blended on stitch)
  → Modified 3D U-Net (residual blocks, dropout; Dice + focal loss; grouped CV)
  → Visualization (slice overlays)
  → Quantitative metrics (Dice, F1, IoU, vesicle density maps)
```

v0 runs the same *shape* of pipeline on **tiny synthetic volumes** so CI stays offline and CPU-minutes. It is not an HPC job and does not claim lab-scale throughput.

## 2. Goals (v0)

1. Deterministic **synthetic ssEM-like volumes** with vesicle-scale blobs (documented voxel size, 40–60 nm diameters, clustered “pools” plus distractor mitochondria-like ovals).
2. Persist volumes in a **Zarr group** (`raw`, `labels`, `instances` + physical metadata).
3. **Adaptive overlapping chunk** planner that avoids sliver tiles and stitches predictions without writing the halo.
4. **Grouped train/val/test split by `volume_id`** with leakage checks (id + raw-volume hash). Never split by random voxel or adjacent section.
5. Train a **tiny residual 3-D U-Net** (dropout; Dice + focal) and a **threshold + morphology baseline**.
6. Report **Dice, F1, IoU, pixel accuracy, vesicle density** (count / µm³) plus overlay and density-map figures under `reports/`.
7. CLI (`generate`, `train`, `infer`, `demo`) and offline GitHub Actions smoke.

## 3. Non-goals (v0)

- Shipping Harris Lab EM, annotations, or any number derived from them (including 88–92% / “35% better than threshold”).
- Claiming clinical or connectomics production performance.
- Full instance-segmentation post (watershed / mutex / affinity graphs). v0 is **semantic** vesicle vs background; instance IDs exist only so density can be counted.
- Large 3-D models, GPU training, architecture search, or real TACC/HPC scheduling.
- Copying code from mentor or lab repositories (see §8).

## 4. Physical / dataset contract

Every volume is a Zarr group:

| Array / attr | Meaning |
|--------------|---------|
| `raw` | `uint8` `(D, H, W)` — simulated EM intensity |
| `labels` | `uint8` `(D, H, W)` — `1` = vesicle, `0` = background |
| `instances` | `uint16` `(D, H, W)` — connected vesicle IDs (0 = background) |
| `voxel_nm` | `(z, y, x)` nanometers per voxel. Default `(40, 4, 4)` — a coarse ssEM-like anisotropy (thick Z, ~2–4 nm-class XY). |
| `volume_id` | stable string; **split unit** |
| `source` | always `synthetic` in v0 |
| `n_vesicles` | placed instance count (simulator ground truth) |

### 4.1 Why these numbers

Synaptic vesicles are ~40–60 nm. At 4 nm/px XY they span ~10–15 pixels; at 40 nm/section they occupy **1–2 slices**. That is the scale the generator uses. Default CI crops are **tiny** (e.g. `8 × 64 × 64` voxels ≈ 0.32 × 0.26 × 0.26 µm) so a residual 3-D net fits in CPU CI. They are **not** Harris Lab tiles.

### 4.2 Simulator cues (not biology)

| Object | Generator behavior |
|--------|--------------------|
| Cytoplasm | Mid-gray, 1/f-ish texture |
| Membranes | Dark curved ridges (threshold bait) |
| Vesicle | Dark rim, paler lumen, clustered at a few “active-zone” sites |
| Mitochondrion-like oval | Larger, darker, elongated — **not labeled**; baseline often fires on these |

A model that scores well here has learned **simulator cues**. That is useful for testing the pipeline. It is not evidence of vesicle detection in real neuropil.

## 5. Split and leakage

- Split is **grouped by `volume_id`**. Default 70 / 15 / 15 train / val / test (CI uses a smaller draw).
- Leakage checker fails the run if:
  - any `volume_id` appears in more than one split
  - any exact raw-volume SHA-256 appears in more than one split
- Chunking a volume for training **must** keep all chunks of that volume in the same split. The API groups on `volume_id` so this is automatic.

Poster “CV” is implemented as this grouped hold-out (plus an optional k-fold helper). We do **not** pretend a 5-fold on 6 synthetic cubes equals the poster’s lab CV.

## 6. Chunking

`vesicle_seg.chunking` plans a grid of tiles of size `chunk` with `overlap` halo on each side.

- Interior writes use only the **inner** (non-halo) region so seams do not double-count.
- **Adaptive last tile:** if the leftover extent along an axis is smaller than `min_edge_frac * chunk`, the previous tile is extended instead of emitting a sliver. This is the v0 stand-in for the poster’s “adaptive chunking” — reduce boundary artifacts without a full HPC block scheduler.

## 7. Models and losses

| Id | Input | Role |
|----|-------|------|
| `threshold` | raw volume | Otsu (or fixed) + opening + size gate. Floor. |
| `resunet3d` | z-scored chunk `(1, D, H, W)` | Tiny residual 3-D U-Net, 8→16→32, `Dropout3d`, bilinear/trilinear upsample |

Loss (poster-aligned):

```
L = dice(p, y) + λ · focal(p, y)
```

with λ default `0.5`, focal γ = `2.0`. Metrics are reported on the **stitched full volume**, not on individual chunks.

## 8. Related public tooling (cited, not copied)

Mentor GitHub: [yajivunev](https://github.com/yajivunev). Used as **problem-shape reference only**. No files were cloned into this tree.

| Public repo | Why it is mentioned | What we did instead |
|-------------|---------------------|---------------------|
| [yajivunev/volara](https://github.com/yajivunev/volara) (upstream style: [e11bio/volara](https://github.com/e11bio/volara)) | Block-wise ops on large nD / Zarr volumes | Original chunk planner + Zarr writer in this repo |
| [yajivunev/missionEM_possible](https://github.com/yajivunev/missionEM_possible) | EM train / predict / IoU eval layout | Original U-Net, trainer, IoU — different package layout |
| [yajivunev/EMISAC](https://github.com/yajivunev/EMISAC) | EM stack artifact context | Contrast jitter in the simulator only; no artifact-correction port |

Harris Lab publications on sparse-annotation 3-D EM (e.g. Venu, Sheridan, Harris, Manor) motivate *why* vesicles and neuropil are hard. They are **not** training data and are **not** re-implemented here.

## 9. Metrics JSON contract

`reports/metrics.json` is produced only by running the code. Required keys:

- `source`: `"synthetic"`
- `seed`, `n_volumes`, `shape`, `voxel_nm`
- `leakage_ok`: boolean
- `poster_draft_note`: string stating 88–92% / 35% figures are **unavailable without lab volumes**
- `models.threshold` / `models.resunet3d`: `{dice, iou, f1, pixel_acc, density_per_um3, density_abs_err}`

No key may contain a hardcoded lab accuracy.

## 10. Demo bar

`make demo` (CI config) should finish in about a minute on CPU and write overlays + `reports/metrics.json` a recruiter can open. See README for the 60–90 s spoken script.
