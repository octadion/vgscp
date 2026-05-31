"""Disk cache for the PRECOMPUTE-ONCE, CACHE-EVERYTHING design (Section 13).

Every frozen-model output is computed once over all splits and cached here as .npy (arrays) +
.json (metadata / variable-length structures). All downstream signals, conformal variants,
budgets and ablations are then cheap vectorized ops over these caches — and, crucially, every
method sees IDENTICAL inputs, making the comparison fair by construction.

Cache key layout:  <root>/<dataset>/<seed>/<split>/<artifact>.npy
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np


class CacheStore:
    def __init__(self, root: str, dataset: str, seed: int):
        self.base = os.path.join(root, dataset, f"seed{seed}")
        os.makedirs(self.base, exist_ok=True)

    def _path(self, split: str, artifact: str, ext: str) -> str:
        d = os.path.join(self.base, split)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{artifact}.{ext}")

    def exists(self, split: str, artifact: str) -> bool:
        return os.path.exists(self._path(split, artifact, "npy")) or os.path.exists(
            self._path(split, artifact, "json")
        )

    def save_array(self, split: str, artifact: str, arr: np.ndarray) -> str:
        path = self._path(split, artifact, "npy")
        np.save(path, np.asarray(arr))
        return path

    def load_array(self, split: str, artifact: str) -> np.ndarray:
        return np.load(self._path(split, artifact, "npy"), allow_pickle=False)

    def save_json(self, split: str, artifact: str, obj: Any) -> str:
        path = self._path(split, artifact, "json")
        with open(path, "w") as f:
            json.dump(obj, f)
        return path

    def load_json(self, split: str, artifact: str) -> Any:
        with open(self._path(split, artifact, "json")) as f:
            return json.load(f)

    def load_all(self, split: str) -> dict:
        """Load every cached artifact for a split into a dict (arrays + json structures)."""
        d = os.path.join(self.base, split)
        out: dict[str, Any] = {}
        if not os.path.isdir(d):
            return out
        for fn in os.listdir(d):
            name, ext = os.path.splitext(fn)
            if ext == ".npy":
                out[name] = np.load(os.path.join(d, fn))
            elif ext == ".json":
                with open(os.path.join(d, fn)) as f:
                    out[name] = json.load(f)
        return out


# Canonical artifact names cached per split (single source of truth).
ARTIFACTS = [
    "logits",            # (N, C)
    "probs",             # (N, C)
    "features",          # (N, d) penultimate phi(x) for trust
    "y_true",            # (N,)
    "y_pred",            # (N,)
    "group_id",          # (N,)
    "spurious_attr",     # (N,)
    "is_minority",       # (N,)
    "member_probs",      # (M, N, C) ensemble
    "mc_pass_probs",     # (K, N, C) mc-dropout
    "concepts",          # (N, concept_dim)
    "pA_given_SM",       # (N, C)
    "pA_given_SA",       # (N, C)
    "reject_prob",       # (N,)
    # variable-length / json:
    "merlin_concepts",   # list[list[int]]
    "morgana_concepts",  # list[list[int]]
]
