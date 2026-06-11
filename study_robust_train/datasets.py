"""Build a backbone-agnostic GridData (train / reweight / eval_domain) for one (dataset, backbone).

Decoupled from the CUB-specific load_real_bundle (AUDIT §5): uses only the binary task label y,
the spurious attribute, and the 4-group id from data.waterbirds / data.celeba (paths-only), plus a
frozen backbone from study_robust_train/features.py. In-domain by construction (every split is the
same composited distribution). Colab-driven (real features); the assembly logic is dataset/backbone
generic.

Split mapping (spec §2 / §3):
  train       = native train split        -> ERM / GroupDRO / balanced subsample fit here
  reweight    = d_learn (held-out)         -> DFR / AFR fit here (textbook reweighting split)
  eval_domain = d_cal + d_test (pooled)    -> conformal cal/test pools resampled from here
"""
from __future__ import annotations

import numpy as np

from data.base import SplitSpec
from .features import clip_features, l2_normalize, resnet50_erm_features
from .grid import GridData


def _load_bundle(dataset: str, cfg: dict, seed: int):
    if dataset == "waterbirds":
        from data.waterbirds import load_waterbirds
        return load_waterbirds(cfg["dataset"], seed, split_spec=SplitSpec(), build_datasets=False)
    if dataset == "celeba":
        from data.celeba import load_celeba
        return load_celeba(cfg["dataset"], seed, split_spec=SplitSpec(), build_datasets=False)
    raise ValueError(f"unknown dataset {dataset!r}")


def _features_for(backbone: str, paths_by_split: dict, y_by_split: dict, cfg: dict, dataset: str) -> dict:
    if backbone == "clip_vitb32":
        clipcfg = cfg.get("clip", {})
        return {sp: l2_normalize(clip_features(
                    paths_by_split[sp], model_name=clipcfg.get("model_name", "ViT-B-32"),
                    pretrained=clipcfg.get("pretrained", "openai"),
                    device=clipcfg.get("device", "cuda"),
                    cache_dir=clipcfg.get("cache_dir", "results/cache_clip"),
                    tag=f"{dataset}_{sp}"))
                for sp in paths_by_split}
    if backbone == "resnet50_erm":
        rcfg = cfg.get("resnet", {})
        return resnet50_erm_features(paths_by_split, y_by_split, tag=dataset,
                                     device=rcfg.get("device", "cuda"),
                                     epochs=rcfg.get("epochs", 10), lr=rcfg.get("lr", 1e-3),
                                     batch_size=rcfg.get("batch_size", 128),
                                     cache_dir=rcfg.get("cache_dir", "results/cache_resnet"))
    raise ValueError(f"unknown backbone {backbone!r}")


def build_griddata(dataset: str, backbone: str, cfg: dict, seed: int = 0) -> GridData:
    """Assemble a GridData for (dataset, backbone). Asserts in-domain: all splits share the
    composited distribution; group = 2*y + spurious."""
    b = _load_bundle(dataset, cfg, seed)
    paths = {sp: b.meta["paths"][sp] for sp in ("train", "d_learn", "d_cal", "d_test")}
    y = {sp: np.asarray(b.y[sp]).astype(int) for sp in paths}
    grp = {sp: np.asarray(b.group_id[sp]).astype(int) for sp in paths}
    feats = _features_for(backbone, paths, y, cfg, dataset)

    eval_X = np.concatenate([feats["d_cal"], feats["d_test"]], axis=0)
    eval_y = np.concatenate([y["d_cal"], y["d_test"]])
    eval_g = np.concatenate([grp["d_cal"], grp["d_test"]])
    return GridData(
        backbone=backbone, dataset=dataset,
        train=(feats["train"], y["train"], grp["train"]),
        reweight=(feats["d_learn"], y["d_learn"], grp["d_learn"]),
        eval_domain=(eval_X, eval_y, eval_g),
        n_classes=int(b.n_classes),
    )
