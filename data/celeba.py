"""CelebA (Blond x Male) loader for the group-robustness study.

Target y = Blond_Hair (1 if blond), spurious attribute = Male (1 if male). group_id = 2*y + male
(the 4 Waterbirds-style groups; blond males are the rare group). Mirrors data/waterbirds.py: parses
CelebA's ``list_attr_celeba.txt`` + ``list_eval_partition.txt`` into paths/y/spurious/group per
split, reusing the shared SplitSpec / split_indices machinery.

The metadata parsing (``parse_attr_file`` / ``parse_partition_file``) is pure and unit-tested on a
tiny fixture (tests/test_celeba_parse.py); the image-encoding step is done by the backbone in
study_robust_train/features.py, so this loader runs paths-only (``build_datasets=False``).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .base import (ImageDatasetBundle, SplitSpec, make_group_id, make_minority_mask,
                   split_indices)

PHASE = 2
N_SPURIOUS_VALUES = 2  # male in {0, 1}


def _find(root: str, name: str) -> Optional[str]:
    for dirpath, _dirs, files in os.walk(root):
        if name in files:
            return os.path.join(dirpath, name)
    return None


def parse_attr_file(path: str) -> tuple[list[str], list[str], np.ndarray]:
    """Parse ``list_attr_celeba.txt`` -> (filenames, attr_names, attr_matrix in {0,1}).

    Format: line 1 = count; line 2 = space-separated attribute names; each subsequent line =
    ``filename v1 ... v40`` with v in {-1, 1}. Returned matrix maps -1->0, 1->1.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    # line 0 may be an integer count; detect and skip if so
    start = 0
    if lines[0].strip().isdigit():
        start = 1
    attr_names = lines[start].split()
    rows, files = [], []
    for ln in lines[start + 1:]:
        parts = ln.split()
        if not parts:
            continue
        files.append(parts[0])
        rows.append([1 if v == "1" else 0 for v in parts[1:]])
    M = np.array(rows, dtype=np.int64)
    return files, attr_names, M


def parse_partition_file(path: str) -> dict:
    """Parse ``list_eval_partition.txt`` -> {filename: split_int (0 train / 1 val / 2 test)}."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f.read().splitlines():
            parts = ln.split()
            if len(parts) >= 2:
                out[parts[0]] = int(parts[1])
    return out


def load_celeba(cfg: dict, seed: int, split_spec: Optional[SplitSpec] = None,
                max_per_split: Optional[int] = None, build_datasets: bool = False) -> ImageDatasetBundle:
    """Build the CelebA bundle (paths/y/spurious/group per split). ``build_datasets`` must be
    False here — the backbone in study_robust_train/features.py encodes images from the paths."""
    if build_datasets:
        raise NotImplementedError("CelebA loader is paths-only; encode via study_robust_train/features.py")
    root = cfg["root"]
    spec = split_spec or SplitSpec()

    attr_path = _find(root, "list_attr_celeba.txt")
    part_path = _find(root, "list_eval_partition.txt")
    if attr_path is None or part_path is None:
        raise FileNotFoundError(
            f"CelebA list_attr_celeba.txt / list_eval_partition.txt not found under {root}. "
            f"Set CELEBA_ROOT/dataset.root to the extracted CelebA.")
    img_dir = _find(root, "000001.jpg")
    img_root = os.path.dirname(img_dir) if img_dir else os.path.join(os.path.dirname(attr_path), "img_align_celeba")

    files, attr_names, M = parse_attr_file(attr_path)
    partition = parse_partition_file(part_path)
    ai = {a: i for i, a in enumerate(attr_names)}
    y_all = M[:, ai["Blond_Hair"]].astype(np.int64)
    male_all = M[:, ai["Male"]].astype(np.int64)
    paths_all = np.array([os.path.join(img_root, fn) for fn in files])
    native_split = np.array([partition.get(fn, 0) for fn in files], dtype=np.int64)

    train_idx = np.where(native_split == 0)[0]
    pool_idx = np.where(native_split != 0)[0]
    idxs = split_indices(len(files), train_idx, pool_idx, spec, seed)

    bundle = ImageDatasetBundle("celeba", n_classes=cfg.get("n_classes", 2))
    bundle.meta["paths"] = {}
    bundle.meta["split_indices"] = {k: v.tolist() for k, v in idxs.items()}
    bundle.meta["img_root"] = img_root
    for split, idx in idxs.items():
        y = y_all[idx]
        male = male_all[idx]
        bundle.y[split] = y
        bundle.spurious_attr[split] = male
        bundle.group_id[split] = make_group_id(y, male, N_SPURIOUS_VALUES)
        bundle.is_minority[split] = make_minority_mask(y, male)
        bundle.meta["paths"][split] = list(paths_all[idx])
    return bundle
