"""Grouped train/val/test split on volume_id + raw-hash leakage checks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vesicle_seg.store.zarr_store import raw_hash
from vesicle_seg.synth.generate import Volume


@dataclass
class SplitResult:
    train: list[str]
    val: list[str]
    test: list[str]
    hashes: dict[str, str] = field(default_factory=dict)
    leakage_ok: bool = True
    notes: list[str] = field(default_factory=list)

    def ids_for(self, split: str) -> list[str]:
        return getattr(self, split)


def grouped_split(
    volumes: list[Volume],
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> SplitResult:
    ids = [v.volume_id for v in volumes]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate volume_id in dataset")
    hashes = {v.volume_id: raw_hash(v.raw) for v in volumes}

    order = np.array(ids)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(order)
    n = len(order)
    n_train = max(1, int(round(fractions[0] * n)))
    n_val = int(round(fractions[1] * n))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1 if n >= 3 else 0
    n_test = n - n_train - n_val
    if n >= 3 and n_test < 1:
        n_train -= 1
        n_test = 1

    train = order[:n_train].tolist()
    val = order[n_train : n_train + n_val].tolist()
    test = order[n_train + n_val :].tolist()
    result = SplitResult(train=train, val=val, test=test, hashes=hashes)
    result.leakage_ok, result.notes = leakage_report(result)
    if not result.leakage_ok:
        raise RuntimeError("leakage detected: " + "; ".join(result.notes))
    return result


def leakage_report(split: SplitResult) -> tuple[bool, list[str]]:
    notes: list[str] = []
    bags = {"train": set(split.train), "val": set(split.val), "test": set(split.test)}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = bags[a] & bags[b]
        if overlap:
            notes.append(f"volume_id overlap {a}/{b}: {sorted(overlap)}")
    hash_owner: dict[str, str] = {}
    for name, ids in bags.items():
        for vid in ids:
            digest = split.hashes.get(vid)
            if digest is None:
                continue
            if digest in hash_owner and hash_owner[digest] != name:
                notes.append(
                    f"raw hash {digest[:12]} in both {hash_owner[digest]} and {name}"
                )
            else:
                hash_owner[digest] = name
    return (len(notes) == 0), notes
