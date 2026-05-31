"""CLEVR-Hans3 loader (Phase-1 PRIMARY real dataset).

CLEVR-Hans3 has 3 classes defined by combinations of object attributes (shape/size/color/
material). The TRAIN split contains a *confounder*: each class co-occurs with a confounding
attribute combination that is broken in the (non-confounded) validation+test splits. The
minority / conflict group is the set of test samples whose confounding attribute disagrees with
the class rule — exactly the case where a shortcut model is confident-but-wrong.

The standard CLEVR-Hans layout is:
    <root>/CLEVR-Hans3/{train,val,test}/images/CLEVR_Hans_classid_*.png
    <root>/CLEVR-Hans3/{train,val,test}/scenes/CLEVR_Hans_scenes_*.json   # per-object attrs
The scene JSON gives per-object ground-truth concepts (shape/size/color/material/position),
which the NCV concept extractor / reimplementation consumes directly.

torch / torchvision are imported lazily so the theory testbed and unit tests run without them.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Optional

import numpy as np

from .base import ImageDatasetBundle, SplitSpec, make_group_id, make_minority_mask, split_indices

# CLEVR-Hans3 class rules (per Stammer et al., 2021). Encoded so we can detect the confounder.
# Each class is defined by a logical rule over object attributes; the train split adds a
# confounding attribute that is predictive in train but not in val/test.
CLASS_RULES = {
    0: "large cube (gray) AND large cylinder",      # confounder: color gray on the cube
    1: "small metal cube AND small sphere",          # confounder: material
    2: "large blue sphere AND small yellow sphere",  # confounder: co-occurring color
}


def _load_scenes(scenes_dir: str) -> list[dict]:
    files = sorted(glob.glob(os.path.join(scenes_dir, "*.json")))
    scenes = []
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        scenes.extend(data.get("scenes", []) if "scenes" in data else [data])
    return scenes


def _concepts_from_scene(scene: dict, max_objects: int = 10) -> np.ndarray:
    """Flatten per-object attributes into a fixed-size ground-truth concept vector.

    Attribute vocab is small and fixed; we one-hot each object's (shape,size,color,material)
    and zero-pad to max_objects. This is the concept space the verifier operates in.
    """
    vocab = {
        "shape": ["cube", "sphere", "cylinder"],
        "size": ["small", "large"],
        "color": ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"],
        "material": ["rubber", "metal"],
    }
    per_obj = sum(len(v) for v in vocab.values())
    out = np.zeros((max_objects, per_obj), dtype=np.float32)
    for i, obj in enumerate(scene.get("objects", [])[:max_objects]):
        off = 0
        for attr, names in vocab.items():
            val = obj.get(attr)
            if val in names:
                out[i, off + names.index(val)] = 1.0
            off += len(names)
    return out.reshape(-1)


def _build_torch_dataset(image_paths, labels, image_size):
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
            return len(image_paths)

        def __getitem__(self, i):
            img = Image.open(image_paths[i]).convert("RGB")
            return tfm(img), int(labels[i])

    return _DS()


def _infer_label_from_filename(path: str) -> int:
    """CLEVR-Hans filenames embed the class id: CLEVR_Hans_classid_<split>_<n>.png."""
    base = os.path.basename(path)
    parts = base.replace(".png", "").split("_")
    for p in parts:
        if p.isdigit() and int(p) < 100:  # class id is small
            return int(p)
    raise ValueError(f"cannot parse class id from {base}")


def load_clevr_hans3(cfg: dict, seed: int, split_spec: Optional[SplitSpec] = None) -> ImageDatasetBundle:
    """Build the CLEVR-Hans3 bundle with group labels and ground-truth concepts.

    Expects ``cfg['root']`` to point at the extracted CLEVR-Hans3 directory. The native train
    split fits f/NCV/ensembles; the native val+test pool is partitioned into d_learn/d_cal/
    d_test (disjoint, seeded). The confounder/minority flag is derived from the scene concepts
    vs the class rule.
    """
    root = cfg["root"]
    image_size = cfg.get("image_size", 224)
    spec = split_spec or SplitSpec()

    bundle = ImageDatasetBundle("clevr_hans3", n_classes=cfg.get("n_classes", 3))

    base = os.path.join(root)
    native = {}
    for split in ("train", "val", "test"):
        img_dir = os.path.join(base, split, "images")
        scn_dir = os.path.join(base, split, "scenes")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(
                f"CLEVR-Hans3 split '{split}' not found at {img_dir}. Set CLEVR_HANS3_ROOT "
                f"or dataset.root to the extracted dataset."
            )
        paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        labels = np.array([_infer_label_from_filename(p) for p in paths], dtype=np.int64)
        scenes = _load_scenes(scn_dir) if os.path.isdir(scn_dir) else [None] * len(paths)
        concepts = np.stack([_concepts_from_scene(s) if s else np.zeros(1) for s in scenes])
        # spurious attr = presence of the train-time confounder concept for the sample's class.
        # Detected from scene attributes; in val/test the confounder is broken => attr != class.
        spurious = _confounder_present(scenes, labels)
        native[split] = {
            "paths": paths, "labels": labels, "concepts": concepts, "spurious": spurious,
        }

    # native train -> 'train'; val+test pool -> d_learn/d_cal/d_test
    all_paths = native["train"]["paths"] + native["val"]["paths"] + native["test"]["paths"]
    all_labels = np.concatenate([native[s]["labels"] for s in ("train", "val", "test")])
    all_conc = np.concatenate([native[s]["concepts"] for s in ("train", "val", "test")])
    all_spur = np.concatenate([native[s]["spurious"] for s in ("train", "val", "test")])
    n_train = len(native["train"]["paths"])
    n_total = len(all_paths)
    train_idx = np.arange(n_train)
    pool_idx = np.arange(n_train, n_total)
    idxs = split_indices(n_total, train_idx, pool_idx, spec, seed)

    n_spur_vals = int(all_spur.max()) + 1 if all_spur.size else 2
    for split, idx in idxs.items():
        y = all_labels[idx]
        sp = all_spur[idx]
        bundle.y[split] = y
        bundle.spurious_attr[split] = sp
        bundle.group_id[split] = make_group_id(y, sp, n_spur_vals)
        bundle.is_minority[split] = make_minority_mask(y, sp)
        bundle.concepts[split] = all_conc[idx]
        bundle.datasets[split] = _build_torch_dataset(
            [all_paths[i] for i in idx], y, image_size
        )
    bundle.meta["split_indices"] = {k: v for k, v in idxs.items()}
    bundle.meta["class_rules"] = CLASS_RULES
    return bundle


def _confounder_present(scenes, labels) -> np.ndarray:
    """Heuristic confounder detector from scene attributes.

    Returns a binary 'spurious attribute' per sample: 1 if the class's train-time confounding
    attribute is present, else 0. In the train split this equals the label-consistent value
    (confounded); in val/test it is broken for the conflict group. For a precise reproduction,
    replace with the official CLEVR-Hans confounder masks if available.
    """
    out = np.zeros(len(labels), dtype=np.int64)
    for i, (scene, y) in enumerate(zip(scenes, labels)):
        if scene is None:
            out[i] = int(y)  # fallback: assume confounded
            continue
        colors = [o.get("color") for o in scene.get("objects", [])]
        materials = [o.get("material") for o in scene.get("objects", [])]
        if y == 0:
            out[i] = int("gray" in colors)
        elif y == 1:
            out[i] = int("metal" in materials)
        else:
            out[i] = int("blue" in colors)
    return out
