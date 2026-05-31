"""Waterbirds loader (critical-path kill-switch testbed).

Waterbirds places land/water birds on land/water backgrounds; the background is the spurious
feature and the worst (minority/conflict) group is bird-on-opposite-background. The standard
release ships a ``metadata.csv`` with (img_filename, y, split, place, place_filename) per image:
  y     : bird type   landbird=0 / waterbird=1
  place : background   land=0 / water=1
  split : 0=train, 1=val, 2=test
  group_id = 2*y + place ; minority/conflict = (y != place)

The native train split fits f / ensemble / verifier; the val+test pool is carved into
d_learn / d_cal / d_test with the shared SplitSpec + split_indices (disjoint, seeded, logged,
disjointness asserted). Concepts are NOT populated here — they come from the frozen CLIP
extractor in precompute (``concept_source: clip``). Never ground-truth attributes (no oracle).

torch / torchvision / PIL are imported lazily so metadata-only use (and the probe's CLIP path,
which loads PIL directly) needs no torch.
"""
from __future__ import annotations

import os
import tarfile
import urllib.request
from typing import Optional

import numpy as np
import pandas as pd

from .base import (
    ImageDatasetBundle,
    SplitSpec,
    make_group_id,
    make_minority_mask,
    split_indices,
)
from conformal.validity import assert_disjoint_splits

PHASE = "A0/A1"
N_PLACE_VALUES = 2  # land / water


def _find_metadata(root: str) -> Optional[str]:
    """Locate metadata.csv under root (the tarball extracts to a nested dir)."""
    for dirpath, _, files in os.walk(root):
        if "metadata.csv" in files:
            return os.path.join(dirpath, "metadata.csv")
    return None


def _download_and_extract(url: str, root: str) -> None:
    os.makedirs(root, exist_ok=True)
    tar_path = os.path.join(root, os.path.basename(url))
    if not os.path.exists(tar_path):
        print(f"[waterbirds] downloading {url} -> {tar_path} (~1.5 GB, one time)")
        urllib.request.urlretrieve(url, tar_path)
    print(f"[waterbirds] extracting {tar_path}")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(root)


def _build_resnet_dataset(paths, labels, image_size):
    """torch Dataset returning (ResNet-normalized tensor, label) — for f/ensemble/MC-dropout."""
    import torch
    from PIL import Image
    from torchvision import transforms

    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            img = Image.open(paths[i]).convert("RGB")
            return tfm(img), int(labels[i])

    return _DS()


def _subsample(idx: np.ndarray, max_n: Optional[int], y: np.ndarray, seed: int) -> np.ndarray:
    """Class-stratified subsample to at most max_n indices (probe speed). Deterministic."""
    if max_n is None or len(idx) <= max_n:
        return idx
    rng = np.random.default_rng(seed)
    per_class = max(1, max_n // len(np.unique(y[idx])))
    keep = []
    for c in np.unique(y[idx]):
        cidx = idx[y[idx] == c]
        rng.shuffle(cidx)
        keep.append(cidx[:per_class])
    out = np.concatenate(keep)
    rng.shuffle(out)
    return out[:max_n]


def load_waterbirds(
    cfg: dict,
    seed: int,
    split_spec: Optional[SplitSpec] = None,
    max_per_split: Optional[int] = None,
    build_datasets: bool = True,
) -> ImageDatasetBundle:
    """Build the Waterbirds bundle with group labels + image paths.

    ``max_per_split`` (probe) class-stratified-subsamples each split for speed. ``build_datasets``
    can be False to skip torch Dataset construction (metadata/paths-only, e.g. the CLIP probe).
    """
    root = cfg["root"]
    image_size = cfg.get("image_size", 224)
    spec = split_spec or SplitSpec()

    meta_path = _find_metadata(root)
    if meta_path is None:
        if cfg.get("download", False):
            _download_and_extract(cfg["url"], root)
            meta_path = _find_metadata(root)
        if meta_path is None:
            raise FileNotFoundError(
                f"Waterbirds metadata.csv not found under {root}. Set WATERBIRDS_ROOT/dataset.root "
                f"to the extracted dataset, or set dataset.download=true."
            )
    img_root = os.path.dirname(meta_path)
    df = pd.read_csv(meta_path)

    paths_all = np.array([os.path.join(img_root, fn) for fn in df["img_filename"]])
    y_all = df["y"].to_numpy().astype(np.int64)
    place_all = df["place"].to_numpy().astype(np.int64)
    native_split = df["split"].to_numpy().astype(np.int64)  # 0 train,1 val,2 test

    train_idx = np.where(native_split == 0)[0]
    pool_idx = np.where(native_split != 0)[0]
    idxs = split_indices(len(df), train_idx, pool_idx, spec, seed)

    # probe subsample (per split, stratified by y)
    if max_per_split is not None:
        idxs = {k: _subsample(v, max_per_split, y_all, seed + i)
                for i, (k, v) in enumerate(idxs.items())}

    assert_disjoint_splits(idxs)

    bundle = ImageDatasetBundle("waterbirds", n_classes=cfg.get("n_classes", 2))
    bundle.meta["paths"] = {}
    bundle.meta["split_indices"] = {k: v.tolist() for k, v in idxs.items()}
    bundle.meta["img_root"] = img_root
    for split, idx in idxs.items():
        y = y_all[idx]
        place = place_all[idx]
        bundle.y[split] = y
        bundle.spurious_attr[split] = place
        bundle.group_id[split] = make_group_id(y, place, N_PLACE_VALUES)
        bundle.is_minority[split] = make_minority_mask(y, place)
        bundle.meta["paths"][split] = list(paths_all[idx])
        if build_datasets:
            bundle.datasets[split] = _build_resnet_dataset(list(paths_all[idx]), y, image_size)
        # NOTE: bundle.concepts is intentionally left empty — CLIP fills it in precompute.
    return bundle
