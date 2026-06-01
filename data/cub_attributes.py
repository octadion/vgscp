"""CUB-200-2011 per-image attribute concepts joined to Waterbirds (probe concept source).

Waterbirds is built from CUB-200-2011, which ships 312 MTurk *per-image* attribute labels
(e.g. ``has_bill_shape::hooked``, ``has_wing_color::black``). These describe the BIRD and are
independent of the pasted Waterbirds background, so (hypothesis) they carry minority bird-type
signal WITHOUT being background-contaminated — unlike the frozen-CLIP global embedding, which is
dominated by the background. This module downloads CUB, parses the per-image attributes, and joins
them to a Waterbirds bundle's image paths to produce an aligned ``(N, 312)`` concept matrix.

CRITICAL honesty constraint (no oracle):
  - Uses PER-IMAGE labels from ``attributes/image_attribute_labels.txt`` only.
  - FORBIDDEN: ``class_attribute_labels_continuous.txt`` / any class-level / majority-vote
    attributes — those are identical for all images of a class, i.e. a deterministic function of
    the label, which would inject the label into the concept space (the oracle trap).
  - Concepts are FROZEN annotations (no training). The bird CLASS label is never used to build
    them.

Only stdlib + numpy are needed here (CPU-only); torch/CLIP are NOT imported.
"""
from __future__ import annotations

import os
import tarfile
import urllib.request
from typing import Optional

import numpy as np

N_ATTRIBUTES = 312
DEFAULT_CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"


# ----------------------------------------------------------------------------------------
# Download / locate
# ----------------------------------------------------------------------------------------
def _download_and_extract(url: str, root: str) -> None:
    os.makedirs(root, exist_ok=True)
    tar_path = os.path.join(root, os.path.basename(url) or "CUB_200_2011.tgz")
    if not os.path.exists(tar_path):
        print(f"[cub] downloading {url} -> {tar_path} (~1.1 GB, one time)")
        urllib.request.urlretrieve(url, tar_path)
    print(f"[cub] extracting {tar_path}")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(root)


def _find_cub_data_root(root: str) -> Optional[str]:
    """Locate the directory that contains ``images.txt`` (the CUB_200_2011 data root)."""
    for dirpath, _, files in os.walk(root):
        if "images.txt" in files and os.path.isdir(os.path.join(dirpath, "images")):
            return dirpath
    # fall back: any dir holding images.txt
    for dirpath, _, files in os.walk(root):
        if "images.txt" in files:
            return dirpath
    return None


def prepare_cub(cub_root: str, download: bool = False, url: Optional[str] = None) -> str:
    """Ensure CUB is extracted under ``cub_root`` and return the data root (holds images.txt)."""
    data_root = _find_cub_data_root(cub_root)
    if data_root is None and download:
        _download_and_extract(url or DEFAULT_CUB_URL, cub_root)
        data_root = _find_cub_data_root(cub_root)
    if data_root is None:
        raise FileNotFoundError(
            f"CUB images.txt not found under {cub_root}. Point cub.root at an extracted "
            f"CUB_200_2011, or set cub.download=true."
        )
    return data_root


# ----------------------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------------------
def _suffix_key(path: str) -> str:
    """Normalized 'species/filename.jpg' key (last two path components), OS-agnostic."""
    parts = path.replace("\\", "/").rstrip("/").split("/")
    return "/".join(parts[-2:]).lower()


def parse_images(data_root: str) -> dict[str, int]:
    """images.txt -> {suffix_key 'species/filename.jpg' : image_id}."""
    path = os.path.join(data_root, "images.txt")
    suffix_to_id: dict[str, int] = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            image_id = int(parts[0])
            img_path = " ".join(parts[1:])  # paths have no spaces in CUB, but be safe
            suffix_to_id[_suffix_key(img_path)] = image_id
    return suffix_to_id


def parse_attribute_labels(
    data_root: str, use_certainty: bool = False
) -> tuple[np.ndarray, int]:
    """image_attribute_labels.txt -> dense (max_image_id+1, 312) per-image attribute matrix.

    Each row of the file is ``<image_id> <attribute_id> <is_present> <certainty_id> <time>``.
    The official file has a KNOWN formatting issue (some rows carry extra tokens), so we parse
    robustly: split on whitespace and read only the FIRST 5 tokens; malformed lines are skipped
    and counted. ``use_certainty`` weights present attributes by certainty_id/4 (default: binary
    ``is_present`` — per the task default).

    Returns ``(matrix, n_malformed)`` where ``matrix[image_id, attribute_id-1]`` is the value.
    """
    path = os.path.join(data_root, "attributes", "image_attribute_labels.txt")
    if not os.path.exists(path):
        path = os.path.join(data_root, "image_attribute_labels.txt")

    img_ids: list[int] = []
    attr_ids: list[int] = []
    vals: list[float] = []
    n_malformed = 0
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                n_malformed += 1
                continue
            try:
                iid = int(parts[0])
                aid = int(parts[1])
                present = int(parts[2])
                certainty = int(parts[3])
            except ValueError:
                n_malformed += 1
                continue
            if aid < 1 or aid > N_ATTRIBUTES:
                n_malformed += 1
                continue
            v = float(present)
            if use_certainty and present:
                v = certainty / 4.0  # certainty_id in 1..4
            img_ids.append(iid)
            attr_ids.append(aid)
            vals.append(v)

    if not img_ids:
        raise ValueError(f"No valid attribute rows parsed from {path}")
    img_arr = np.asarray(img_ids, dtype=np.int64)
    attr_arr = np.asarray(attr_ids, dtype=np.int64)
    val_arr = np.asarray(vals, dtype=np.float32)
    max_id = int(img_arr.max())
    matrix = np.zeros((max_id + 1, N_ATTRIBUTES), dtype=np.float32)
    matrix[img_arr, attr_arr - 1] = val_arr
    if n_malformed:
        print(f"[cub] skipped {n_malformed} malformed attribute rows (expected; known file issue)")
    return matrix, n_malformed


def parse_attribute_names(data_root: str) -> list[str]:
    """attributes.txt -> list of 312 attribute names; fall back to integer ids if absent."""
    candidates = [
        os.path.join(data_root, "attributes.txt"),
        os.path.join(data_root, "attributes", "attributes.txt"),
        os.path.join(os.path.dirname(data_root), "attributes.txt"),
    ]
    names_by_id: dict[int, str] = {}
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    try:
                        aid = int(parts[0])
                    except ValueError:
                        continue
                    names_by_id[aid] = " ".join(parts[1:])
            break
    if len(names_by_id) >= N_ATTRIBUTES:
        return [names_by_id.get(i, f"attr_{i}") for i in range(1, N_ATTRIBUTES + 1)]
    print("[cub] attributes.txt not found / incomplete; using integer attribute ids")
    return [f"attr_{i}" for i in range(1, N_ATTRIBUTES + 1)]


# ----------------------------------------------------------------------------------------
# Join to Waterbirds
# ----------------------------------------------------------------------------------------
def load_cub_attribute_concepts(
    bundle,
    cub_root: str,
    splits: tuple[str, ...] = ("train", "d_test"),
    download: bool = False,
    url: Optional[str] = None,
    use_certainty: bool = False,
    min_coverage: float = 0.99,
) -> tuple[dict[str, np.ndarray], list[str], dict]:
    """Join per-image CUB attributes onto a Waterbirds bundle.

    Returns ``(concepts, attr_names, info)`` where ``concepts[split]`` is an ``(N_split, 312)``
    float array aligned row-for-row to ``bundle.meta["paths"][split]``, ``attr_names`` is the
    312-long attribute-name list, and ``info`` records join coverage + any unmatched examples.

    Asserts overall join coverage >= ``min_coverage`` (default 0.99) across the requested splits;
    if coverage is low the path-matching key is wrong and we RAISE rather than silently drop.
    """
    data_root = prepare_cub(cub_root, download=download, url=url)
    suffix_to_id = parse_images(data_root)
    matrix, n_malformed = parse_attribute_labels(data_root, use_certainty=use_certainty)
    attr_names = parse_attribute_names(data_root)

    concepts: dict[str, np.ndarray] = {}
    n_total = 0
    n_matched = 0
    unmatched: list[str] = []
    for split in splits:
        paths = bundle.meta["paths"][split]
        X = np.zeros((len(paths), N_ATTRIBUTES), dtype=np.float32)
        matched = np.zeros(len(paths), dtype=bool)
        for i, p in enumerate(paths):
            key = _suffix_key(p)
            image_id = suffix_to_id.get(key)
            if image_id is not None and image_id < matrix.shape[0]:
                X[i] = matrix[image_id]
                matched[i] = True
            elif len(unmatched) < 20:
                unmatched.append(p)
        concepts[split] = X
        n_total += len(paths)
        n_matched += int(matched.sum())

    coverage = n_matched / max(1, n_total)
    info = {
        "coverage": float(coverage),
        "n_matched": int(n_matched),
        "n_total": int(n_total),
        "n_malformed_attr_rows": int(n_malformed),
        "unmatched_examples": unmatched,
        "use_certainty": bool(use_certainty),
        "data_root": data_root,
    }
    print(f"[cub] join coverage: {n_matched}/{n_total} = {coverage:.4f}")
    assert coverage >= min_coverage, (
        f"CUB<->Waterbirds join coverage {coverage:.4f} < {min_coverage}. The path-matching key "
        f"is likely wrong. First unmatched: {unmatched[:5]}"
    )
    return concepts, attr_names, info
