"""Shared REAL-data builder for E1–E4 on FROZEN CLIP features (the §2b runtime trick).

Everything the real experiments need is derived from ONE cached set of frozen CLIP ViT-B/32 image
features, plus the per-image CUB attributes and the Waterbirds metadata -- NO large-model training
(only logistic heads on the cached inputs). This keeps every real run well under the 5h budget.

Pipeline:
  * ``load_real_bundle``     : Waterbirds metadata (paths, binary y, place, group) + per-image CUB
    species id (from the path) + 312 CUB attributes (per-image only; no oracle) + cached CLIP
    image features. Uses the repo's split machinery (train / d_learn / d_cal / d_test).
  * ``clip_image_features``  : encode images ONCE via models.concept_extractor_clip and cache to
    disk (cache key = model + pretrained + the exact path list). Raises a clear error if open_clip
    or the images are missing.
  * ``fit_logistic_head`` / ``head_probs`` : multinomial logistic head on a frozen input, with
    column alignment to the full label space (reuses signals.conformal_scores.probe_posteriors).
  * ``build_binary_fdata``  : assemble the binary-Waterbirds ``fdata`` dict (the structure E2/E3 and
    the kill-switch already consume) entirely from cached features -- f-softmax from a binary head,
    optional logistic-ensemble member probs and a feature-dropout MC proxy for E2's baselines.

These downstream steps are pure numpy/sklearn and are unit-tested with INJECTED synthetic features
(no CLIP/data needed); only the CLIP-encoding + dataset-loading steps require the real environment.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from data.base import SPLITS, SplitSpec
from signals.conformal_scores import probe_posteriors


# ======================================================================================
# CLIP image features (encode once, cache to disk)
# ======================================================================================
def _paths_hash(paths: list[str]) -> str:
    h = hashlib.sha1()
    for p in paths:
        h.update(os.path.basename(p).encode("utf-8", "ignore"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def clip_image_features(paths: list[str], model_name: str, pretrained: str, device: str,
                        cache_dir: str = "results/cache_clip", batch_size: int = 64,
                        tag: str = "") -> np.ndarray:
    """(N, d) L2-normalized frozen CLIP image features for ``paths``; cached to disk.

    Cache key includes model+pretrained+tag+a hash of the (basename-ordered) path list, so a
    re-run with the same images skips re-encoding. Raises a clear error if open_clip / images are
    unavailable (we never fabricate features)."""
    os.makedirs(cache_dir, exist_ok=True)
    key = f"{model_name}_{pretrained}_{tag}_{_paths_hash(paths)}_{len(paths)}".replace("/", "-")
    cache_path = os.path.join(cache_dir, f"clipfeat_{key}.npy")
    if os.path.exists(cache_path):
        feats = np.load(cache_path)
        if feats.shape[0] == len(paths):
            return feats
    try:
        from models.concept_extractor_clip import CLIPConceptExtractor
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"CLIP feature extraction needs open_clip: {e}")
    # concept_bank is irrelevant for image features; pass a dummy single prompt.
    extractor = CLIPConceptExtractor(model_name, pretrained, ["a photo"], device=device)
    feats = extractor.encode_image_features(paths, batch_size=batch_size)
    np.save(cache_path, feats)
    return feats


# ======================================================================================
# Logistic heads on frozen inputs (no large-model training)
# ======================================================================================
def fit_logistic_head(X_train: np.ndarray, y_train: np.ndarray, C: float = 1.0,
                      max_iter: int = 2000, seed: int = 0):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=max_iter, C=C, random_state=seed, n_jobs=None)
    clf.fit(X_train, y_train)
    return clf


def head_probs(clf, X: np.ndarray, n_classes: int) -> np.ndarray:
    """(N, n_classes) class posteriors, scattered to the full label space (missing classes -> 0)."""
    return probe_posteriors(clf, X, n_classes)


def clip_text_concepts(features: np.ndarray, model_name: str, pretrained: str, device: str,
                       prompts: list[str]) -> np.ndarray:
    """(N, K) cosine concept scores = (cached L2-normalized image features) @ (text embeddings)^T.

    Reuses the ALREADY-CACHED image features (no image re-encode) and only loads CLIP's text tower
    to embed the K prompts. Used for E2's spurious scene/background concept channel."""
    try:
        import torch

        from models.concept_extractor_clip import CLIPConceptExtractor
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"CLIP text concepts need open_clip: {e}")
    ext = CLIPConceptExtractor(model_name, pretrained, prompts, device=device)
    ext.load()
    temb = ext._text_emb.float().cpu().numpy()           # (K, d), already L2-normalized
    return (features @ temb.T).astype(np.float32)


# ======================================================================================
# Real Waterbirds + CUB-200 bundle on cached CLIP features
# ======================================================================================
@dataclass
class RealBundle:
    features: dict                 # split -> (N, d) frozen CLIP features
    attrs: dict                    # split -> (N, 312) per-image CUB attributes
    species: dict                  # split -> (N,) 0-based CUB species id
    y: dict                        # split -> (N,) binary waterbird/landbird (the species type)
    place: dict                    # split -> (N,) binary background (land=0/water=1)
    group_id: dict                 # split -> (N,) Waterbirds 4-group id
    is_minority: dict              # split -> (N,) bool  (place != y)
    paths: dict                    # split -> list[str]
    species_type: np.ndarray       # (n_species,) -> binary type t(species) (== y of that species)
    attr_names: list
    n_classes: int                 # species label-space size (default 200)
    info: dict = field(default_factory=dict)


def _species_type_map(species_all: np.ndarray, y_all: np.ndarray, n_classes: int) -> np.ndarray:
    """t(species) in {0,1}: the binary Waterbirds label of that species (constant per species).

    Computed by majority vote per species over all (species, y) pairs (robust to any stray label);
    species never seen default to 0 and are flagged in info by the caller."""
    t = np.zeros(n_classes, dtype=np.int64)
    for s in range(n_classes):
        m = species_all == s
        if m.any():
            t[s] = int(round(float(y_all[m].mean())))
    return t


def load_real_bundle(cfg: dict, seed: int, splits: tuple = SPLITS) -> RealBundle:
    """Load Waterbirds + CUB-200 and the cached CLIP features into a ``RealBundle``.

    Reuses data.waterbirds.load_waterbirds (paths/y/place/group; no torch datasets needed) and
    data.cub_attributes.load_cub_attribute_concepts (per-image 312 attributes). Encodes CLIP image
    features once per split (cached). Raises clearly if datasets / open_clip are missing."""
    from data.cub_attributes import load_cub_attribute_concepts
    from data.waterbirds import load_waterbirds
    from experiments.cub200_frontier import species_from_waterbirds_paths

    dcfg = cfg["dataset"]
    # species label space is ALWAYS the 200 CUB species (independent of the binary-f task that
    # E2/E3 set via dataset.n_classes=2). species ids are 0..199 from the path.
    n_classes = int(cfg.get("cub", {}).get("n_species", 200))
    spec = SplitSpec(**{k: cfg["common"]["splits"][k] for k in ("d_learn", "d_cal", "d_test")}) \
        if "common" in cfg and "splits" in cfg["common"] else SplitSpec()
    bundle = load_waterbirds(dcfg, seed, split_spec=spec, build_datasets=False)

    attrs, attr_names, join_info = load_cub_attribute_concepts(
        bundle, cfg["cub"]["root"], splits=splits, download=cfg["cub"].get("download", False),
        url=cfg["cub"].get("url"), use_certainty=cfg["cub"].get("use_certainty", False),
        min_coverage=cfg["cub"].get("min_coverage", 0.99))

    clipcfg = cfg.get("clip", {})
    model_name = clipcfg.get("model_name", "ViT-B-32")
    pretrained = clipcfg.get("pretrained", "openai")
    device = clipcfg.get("device", "cuda")
    cache_dir = clipcfg.get("cache_dir", "results/cache_clip")

    features, species, y, place, group_id, is_minority, paths = {}, {}, {}, {}, {}, {}, {}
    for sp in splits:
        p = list(bundle.meta["paths"][sp])
        paths[sp] = p
        features[sp] = clip_image_features(p, model_name, pretrained, device, cache_dir, tag=sp)
        species[sp] = species_from_waterbirds_paths(p)
        y[sp] = np.asarray(bundle.y[sp]).astype(np.int64)
        place[sp] = np.asarray(bundle.spurious_attr[sp]).astype(np.int64)
        group_id[sp] = np.asarray(bundle.group_id[sp]).astype(np.int64)
        is_minority[sp] = np.asarray(bundle.is_minority[sp]).astype(bool)

    species_all = np.concatenate([species[s] for s in splits])
    y_all = np.concatenate([y[s] for s in splits])
    species_type = _species_type_map(species_all, y_all, n_classes)
    info = {"cub_join": join_info, "n_species_present": int(len(np.unique(species_all))),
            "attr_dim": int(attrs[splits[0]].shape[1]), "feat_dim": int(features[splits[0]].shape[1])}
    return RealBundle(features=features, attrs={s: attrs[s] for s in splits}, species=species,
                      y=y, place=place, group_id=group_id, is_minority=is_minority, paths=paths,
                      species_type=species_type, attr_names=attr_names, n_classes=n_classes,
                      info=info)


# ======================================================================================
# Binary-Waterbirds fdata on cached features (for E2 / E3, structure-compatible with the repo)
# ======================================================================================
def _ensemble_member_probs(X_train, y_train, X, n_classes, n_members, seed):
    """(M, N, C) logistic-ensemble member probs: M heads on bootstrap resamples of TRAIN."""
    rng = np.random.default_rng(seed)
    out = []
    for m in range(n_members):
        idx = rng.integers(0, len(X_train), len(X_train))
        clf = fit_logistic_head(X_train[idx], y_train[idx], seed=seed + m)
        out.append(head_probs(clf, X, n_classes))
    return np.stack(out, axis=0).astype(np.float32)


def _mc_dropout_proxy(clf, X, n_classes, n_passes, drop_p, seed):
    """(K, N, C) MC-dropout PROXY on frozen features: K passes with random feature dropout +
    rescaling on the input. Not true model dropout (the head is linear), but a torch-free stochastic
    perturbation that yields a usable mcdropout disagreement baseline on cached features."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n_passes):
        mask = (rng.random(X.shape) >= drop_p).astype(X.dtype) / (1.0 - drop_p)
        out.append(head_probs(clf, X * mask, n_classes))
    return np.stack(out, axis=0).astype(np.float32)


def build_binary_fdata(bundle: RealBundle, cfg: dict, seed: int, splits: tuple = SPLITS,
                       with_ensemble_mc: bool = False) -> dict:
    """Assemble the binary-Waterbirds ``fdata`` dict from cached CLIP features.

    f = logistic head (CLIP features -> binary y). ``with_ensemble_mc`` additionally builds the
    logistic-ensemble member probs and the feature-dropout MC proxy (only E2 needs those; E3 does
    not). features = the raw CLIP features (used by the trust signal). Concept attributes are
    carried alongside in ``attrs`` for the concept-space scores / verifier."""
    nC = 2
    Xtr, ytr = bundle.features["train"], bundle.y["train"]
    f_head = fit_logistic_head(Xtr, ytr, C=float(cfg.get("heads", {}).get("C", 1.0)),
                               max_iter=int(cfg.get("heads", {}).get("max_iter", 2000)), seed=seed)
    n_members = int(cfg.get("common", {}).get("ensemble", {}).get("n_members", 5))
    n_passes = int(cfg.get("common", {}).get("mc_dropout", {}).get("n_passes", 20))
    fdata = {}
    for sp in splits:
        X = bundle.features[sp]
        probs = head_probs(f_head, X, nC).astype(np.float32)
        d = {"probs": probs, "y_pred": probs.argmax(1).astype(np.int64),
             "y_true": bundle.y[sp].astype(np.int64), "features": X.astype(np.float32),
             "group_id": bundle.group_id[sp], "spurious_attr": bundle.place[sp],
             "is_minority": bundle.is_minority[sp]}
        if with_ensemble_mc:
            d["member_probs"] = _ensemble_member_probs(Xtr, ytr, X, nC, n_members, seed)
            d["mc_pass_probs"] = _mc_dropout_proxy(f_head, X, nC, n_passes, drop_p=0.2,
                                                   seed=seed + 1)
        fdata[sp] = d
    return fdata
