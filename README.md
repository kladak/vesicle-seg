# vesicle-seg

**Clean-room scientific ML — synaptic-vesicle segmentation on synthetic EM-like volumes.**

Portfolio project for [Karim Ladak](https://github.com/kladak). Spec: [`SPEC.md`](SPEC.md).

Implements the *shape* of a Harris Lab (UT Austin) poster pipeline as **educational research tooling** on a public simulator. Distinct from [MotionCode](https://github.com/kladak/motioncode) (1-D ECG-like classification).

Poster (concepts only): *“Fast vesicle segmentation in neural imaging: An HPC and deep learning approach”* (mentor Vijay Venu).

```text
Raw EM-like volume → Zarr → adaptive overlapping chunks
  → residual 3-D U-Net (dropout; Dice + focal)
  → stitched prediction → overlays + vesicle density maps
```

## Honesty (read first)

- **Educational research tooling.** Not clinical imaging software. Not a medical device. Not for diagnosis or published biological measurements.
- **Synthetic data only** in the default / CI path. No Harris Lab serial-section EM, no lab annotations, no internal code.
- **Not an official Harris Lab release** and not affiliated as one. Independent clean-room implementation.
- A poster draft quotes **~88–92%** accuracy and **“35% better than threshold.”** Those figures are **not reconstructible** without lab volumes. They are **not** written into `reports/metrics.json`. Trust only numbers this repo computes on the simulator.
- Vesicle diameters (~40–60 nm) and voxel size `(40, 4, 4)` nm are **documented simulator defaults**, not a dump of lab metadata.

## Demo script (60–90s)

Cold start for a recruiter walkthrough or screen recording.

### 0. Once

```bash
git clone https://github.com/kladak/vesicle-seg.git
cd vesicle-seg
python -m pip install -e ".[dev]"
# CPU torch is enough. If the wheel is missing: pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 1. Generate the Zarr store (~10s)

```bash
make generate
```

Point at `data/synthetic.zarr`: groups `syn_000`… with `raw`, `labels`, `instances`, `voxel_nm`. This is the poster’s “distributed storage” step, shrunk to a laptop folder.

### 2. Train + infer (~40s)

```bash
make demo          # configs/ci.yaml — residual 3D U-Net + threshold baseline
```

Say out loud:

1. Volumes are **synthetic** ssEM-like cubes (8×64×64). Vesicles are 40–60 nm blobs clustered like pools; mitochondria-like ovals are unlabeled distractors.
2. Split is **by `volume_id`**, with raw-hash leakage checks — we do not randomly split adjacent sections.
3. Inference reads **overlapping chunks** and writes only the inner region so tile seams do not double-count.
4. Open `reports/overlay_panel.png` — raw | GT | threshold | U-Net.
5. Open `reports/metrics.json` — Dice, F1, IoU, density / µm³. **These are simulator scores.** Mention the poster 88–92% only to say it is *not* in this file.

### 3. Optional viewer (~15s)

```bash
make serve         # http://127.0.0.1:8765/viewer/
```

### 4. Close (~10s)

`make test` is what CI runs, plus `make train-ci`. No network. No lab data.

## What you get

| Piece | Path |
|-------|------|
| Spec | [`SPEC.md`](SPEC.md) |
| Experiment YAMLs | [`configs/ci.yaml`](configs/ci.yaml), [`configs/default.yaml`](configs/default.yaml) |
| Synthetic generator | [`vesicle_seg/synth/generate.py`](vesicle_seg/synth/generate.py) |
| Zarr store | [`vesicle_seg/store/zarr_store.py`](vesicle_seg/store/zarr_store.py) |
| Adaptive overlapping chunks | [`vesicle_seg/chunking/tiles.py`](vesicle_seg/chunking/tiles.py) |
| Leakage-aware split | [`vesicle_seg/data/splits.py`](vesicle_seg/data/splits.py) |
| Residual 3-D U-Net + Dice/focal | [`vesicle_seg/models/`](vesicle_seg/models/) |
| Threshold baseline | [`vesicle_seg/models/baseline.py`](vesicle_seg/models/baseline.py) |
| Train / stitched infer | [`vesicle_seg/train.py`](vesicle_seg/train.py), [`vesicle_seg/infer.py`](vesicle_seg/infer.py) |
| CLI | `python -m vesicle_seg.cli {generate,train,infer,demo,serve}` |
| CI | [`.github/workflows/ci.yml`](.github/workflows/ci.yml) |

## Synthetic CI metrics (real run, seed=42)

Produced by `make train-ci` on this machine (2026-09-14). **n_volumes=6**, shape `8×64×64`, voxel `(40, 4, 4)` nm, 8 epochs, leakage_ok=true. These are **simulator** scores, not Harris Lab performance. If this table disagrees with [`reports/metrics.example.json`](reports/metrics.example.json) / a fresh `reports/metrics.json`, **trust the JSON**.

| Model | Dice | IoU | F1 | pixel acc |
|-------|------|-----|----|-----------|
| threshold + morphology | 0.226 | 0.132 | 0.226 | 0.960 |
| resunet3d (8 epochs, CPU) | 0.460 | 0.304 | 0.460 | 0.973 |

On this draw the residual net **beats the threshold floor on Dice / IoU / F1** (high precision, conservative recall). Pixel accuracy is dominated by background and is the wrong headline metric. Density / µm³ is a count on a ~0.021 µm³ crop — a pipeline check, not a published bouton density.

Takeaways that are allowed on a resume:

- Leakage-checked **volume-level** splits and overlapping-chunk stitch are implemented and tested.
- A **threshold + morphology** floor exists so the residual U-Net has something to beat on the *simulator*.
- Absolute Dice/IoU will move with seed and `n_volumes`. **Do not quote poster 88–92% as a result of this repo.**

## Related public EM tooling (cited, not copied)

Mentor GitHub: [yajivunev](https://github.com/yajivunev). Problem-shape reference only — **no files from those repos are in this tree**.

- [volara](https://github.com/yajivunev/volara) / [e11bio/volara](https://github.com/e11bio/volara) — block-wise ops on large nD / Zarr volumes
- [missionEM_possible](https://github.com/yajivunev/missionEM_possible) — EM train / predict / IoU eval *layout*
- [EMISAC](https://github.com/yajivunev/EMISAC) — EM stack artifact context

## Tests

```bash
make test
```

Covers generator determinism, Zarr round-trip, chunk coverage + stitch, id/hash leakage, Dice/IoU identities, U-Net shapes, baseline binary output, and a 1-epoch synthetic smoke train.

## License

MIT — see [`LICENSE`](LICENSE).
