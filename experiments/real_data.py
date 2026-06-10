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


def assert_l2_normalized(X: np.ndarray, tag: str = "features", tol: float = 1e-2) -> None:
    """Guard for confound #1 (run spec v2 §2a): CLIP features MUST be L2-normalized before a linear
    head, or the probe collapses to ~chance. Raises with a clear diagnosis if not unit-norm."""
    norms = np.linalg.norm(np.asarray(X, dtype=np.float64), axis=1)
    if X.shape[0] and not np.allclose(norms, 1.0, atol=tol):
        raise ValueError(
            f"[head-fix §2a] {tag} are NOT L2-normalized (mean‖x‖={norms.mean():.3f}, "
            f"min={norms.min():.3f}, max={norms.max():.3f}). A linear probe on un-normalized CLIP "
            f"features under-fits badly (this was a prime suspect for the 0.182 head). L2-normalize "
            f"the cached features before fitting/scoring.")


# ======================================================================================
# §1 (v3) HARD ACCURACY GATE + standardized linear probe (the head-fix)
# ======================================================================================
GATE_MIN_TOP1 = 0.55             # §1.4: feature head clean-CUB top-1 must clear this (HARD HALT)
BASELINE_MIN_TOP1 = 0.55         # §1.2: standard CLIP linear-probe sanity anchor


class FeatureHeadGateError(RuntimeError):
    """Raised when the feature head fails the §1.4 clean-CUB accuracy gate. The orchestrator catches
    this, writes BLOCKERS_v3.md, and emits NO 2x2 and NO verdict (the v2 run wrongly proceeded)."""
    def __init__(self, top1: float, diagnosis: str, where: str = "study head"):
        self.top1 = float(top1)
        self.diagnosis = diagnosis
        self.where = where
        super().__init__(f"[§1.4 GATE FAILED @ {where}] clean-CUB top-1={top1:.3f} < {GATE_MIN_TOP1}. "
                         f"{diagnosis}")


def fit_species_head(X_train: np.ndarray, y_train: np.ndarray, C: float = 1.0,
                     max_iter: int = 5000, seed: int = 0):
    """STANDARDIZED multinomial logistic head on frozen CLIP features (the §1.1 head-fix).

    The v2 head fit a raw ``LogisticRegression`` directly on unit-norm CLIP features -- the standard
    CLIP-linear-probe pitfall (ill-conditioned per-dim scales + under-convergence -> a ~0.16-accuracy
    probe). The fix is the textbook recipe: z-score the features with TRAIN-only stats, then a
    well-converged multinomial logistic. Returned as a sklearn Pipeline so ``probe_posteriors`` /
    ``predict_proba`` / ``classes_`` work transparently."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # (lbfgs defaults to multinomial for multiclass; don't pass the deprecated multi_class kwarg)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=C, max_iter=max_iter, random_state=seed))
    clf.fit(X_train, y_train)
    return clf


def top1(clf, X: np.ndarray, y: np.ndarray) -> float:
    return float((clf.predict(X) == y).mean())


def clip_linear_probe_baseline(X_train, y_train, X_eval, y_eval, C: float = 1.0,
                               max_iter: int = 5000, seed: int = 0) -> dict:
    """§1.2 KNOWN-GOOD baseline: a standard standardized CLIP linear probe on clean CUB-200. If this
    cannot reach >=0.55 the bug is in feature extraction / label alignment (NOT the study head) --
    the caller STOPS and reports. Returns {top1, passes, n_train, n_eval, n_classes}."""
    clf = fit_species_head(X_train, y_train, C=C, max_iter=max_iter, seed=seed)
    acc = top1(clf, X_eval, y_eval)
    return {"top1": acc, "passes": bool(acc >= BASELINE_MIN_TOP1), "n_train": int(len(y_train)),
            "n_eval": int(len(y_eval)), "n_classes": int(len(np.unique(y_train)))}


def assert_label_alignment(paths_a: list[str], paths_b: list[str], tag: str = "") -> None:
    """§1.1 guard against label/path desync: two path lists that should describe the SAME images
    (row-for-row) must share basenames. Catches a reordered/misjoined cache before it silently
    misaligns features and labels (a prime suspect for a ~chance head)."""
    if len(paths_a) != len(paths_b):
        raise ValueError(f"[label-align {tag}] length mismatch {len(paths_a)} vs {len(paths_b)}")
    mism = [(a, b) for a, b in zip(paths_a, paths_b)
            if os.path.basename(str(a)) != os.path.basename(str(b))]
    if mism:
        raise ValueError(f"[label-align {tag}] {len(mism)} row(s) have mismatched basenames, e.g. "
                         f"{mism[0]} -- the feature cache and labels are DESYNCED.")


# ======================================================================================
# §2b: IMAGE-DERIVED (predicted) concept features -- the corrected concept source
# ======================================================================================
def fit_attribute_probe(X_feats_train: np.ndarray, attrs_train: np.ndarray, C: float = 1.0,
                        max_iter: int = 1000, seed: int = 0):
    """CBM bottleneck probe: frozen CLIP image features -> per-attribute presence probability.

    One logistic regressor per attribute (binary present/absent), fit on TRAIN only. At test time
    we use the PREDICTED attribute probabilities (image-derived), never the ground-truth MTurk
    labels -- this is the §2b leakage fix. Constant attributes (always 0/1 in TRAIN) get a constant
    predictor. Returns a list of (col_index, clf-or-const) we can vectorize over.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    attrs_bin = (np.asarray(attrs_train) >= 0.5).astype(int)
    probes = []
    for j in range(attrs_bin.shape[1]):
        yj = attrs_bin[:, j]
        if yj.min() == yj.max():                 # constant attribute -> constant predictor
            probes.append(("const", float(yj.mean())))
            continue
        # standardized (§1.1 head-fix recipe) so the per-attribute probe doesn't under-fit on
        # ill-conditioned unit-norm CLIP features
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=max_iter, C=C, random_state=seed))
        clf.fit(X_feats_train, yj)
        probes.append(("clf", clf))
    return probes


def predict_attribute_features(probes, X_feats: np.ndarray) -> np.ndarray:
    """(N, n_attr) PREDICTED attribute probabilities from the CBM probe (image-derived, §2b)."""
    out = np.zeros((X_feats.shape[0], len(probes)), dtype=np.float32)
    for j, (kind, obj) in enumerate(probes):
        if kind == "const":
            out[:, j] = obj
        else:
            out[:, j] = obj.predict_proba(X_feats)[:, list(obj.classes_).index(1)]
    return out


def clip_zeroshot_attribute_features(features: np.ndarray, attr_names: list, model_name: str,
                                     pretrained: str, device: str,
                                     templates: Optional[list] = None) -> np.ndarray:
    """(N, n_attr) CLIP ZERO-SHOT attribute scores = cosine(image features, attribute-text embedding).

    Fully image-derived, NO attribute training (the cleanest leakage-free concept source, §2b
    appendix). Each CUB attribute name (e.g. ``has_wing_color::black``) is turned into a readable
    prompt; the score is the cached L2-normalized image feature dotted with the attribute's text
    embedding (reuses ``clip_text_concepts`` -> no image re-encode)."""
    def humanize(a: str) -> str:
        a = a.split("::")
        part = a[0].replace("has_", "").replace("_", " ")
        val = a[1].replace("_", " ") if len(a) > 1 else ""
        return f"a bird with {part} {val}".strip()
    prompts = [humanize(a) for a in attr_names]
    return clip_text_concepts(features, model_name, pretrained, device, prompts)


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
# §2a CLEAN-CUB feature head + §2b predicted-concept E1 population (the CORRECTED real path)
# ======================================================================================
def clean_cub_image_paths(wb_paths: list[str], cub_data_root: str) -> list[str]:
    """Map Waterbirds (background-COMPOSITED) paths -> the ORIGINAL clean CUB-200 image paths.

    Each Waterbirds path ends in ``<species_folder>/<filename>.jpg``; the clean image lives at
    ``<cub_data_root>/images/<species_folder>/<filename>.jpg``. Used so the feature head can be
    trained/evaluated on CLEAN CUB (confound #1: training on composited images degraded the head)."""
    out = []
    for p in wb_paths:
        parts = p.replace("\\", "/").rstrip("/").split("/")
        out.append(os.path.join(cub_data_root, "images", "/".join(parts[-2:])))
    return out


def assemble_e1_population(cfg: dict, seed: int) -> dict:
    """CORRECTED E1/unified population (run spec v2 §2a + §2b). Colab/GPU only.

    Fixes vs the prior run:
      §2a  the FEATURE head is trained on CLEAN CUB-200 features (not background-composited ones)
           and its clean top-1 is reported as the sanity target (>=0.55). Frozen features are
           L2-normalized (asserted) before the linear head -- the prime suspect for the 0.182 bug.
           The frontier still SCORES the contaminated (Waterbirds-composited) pool posteriors, so the
           contamination shows up in the SCORE under shift, not in a broken head.
      §2b  the CONCEPT score is IMAGE-DERIVED (predicted), never the ground-truth MTurk attributes:
             * cbm        : CLIP features -> attribute probe -> predicted attrs -> species head
             * zeroshot   : CLIP zero-shot attribute cosines -> species head
             * gt_attrs_leaky : the PRIOR (invalid) path -- ground-truth attrs at test time. Kept ONLY
               for the leakage demonstration; emits a loud warning and must not be the headline.

    Returns the same dict shape as ``make_smoke_population`` plus ``feat_top1_cleancub`` (the §2a
    sanity number) and ``concept_source``. This needs torch + open_clip + the CUB/Waterbirds data;
    every sub-step except CLIP-encoding is pure numpy/sklearn (unit-testable with injected arrays)."""
    from data.cub_attributes import prepare_cub
    from experiments.cub200_frontier import realized_rho, typicality_group, ATYPICAL, TYPICAL

    concept_source = cfg.get("concept_source", "cbm")
    valid = ("cbm", "zeroshot", "gt_attrs_leaky")
    if concept_source not in valid:
        raise ValueError(f"concept_source {concept_source!r} not in {valid}")
    if concept_source == "gt_attrs_leaky":
        print("[e1 §2b] WARNING: concept_source='gt_attrs_leaky' scores GROUND-TRUTH MTurk "
              "attributes at TEST time. This is the PRIOR, INVALID behaviour (label leakage); use it "
              "ONLY to demonstrate the leak, never as the headline concept result.")

    bundle = load_real_bundle(cfg, seed)
    n_classes = bundle.n_classes
    hcfg = cfg.get("heads", {})
    # v3: standardized head defaults -> higher max_iter; C overridable
    C, max_iter = float(hcfg.get("C", 1.0)), int(hcfg.get("max_iter", 5000))
    pool_splits = tuple(cfg.get("pool_splits", ("d_learn", "d_cal", "d_test")))
    clipcfg = cfg.get("clip", {})
    cargs = (clipcfg.get("model_name", "ViT-B-32"), clipcfg.get("pretrained", "openai"),
             clipcfg.get("device", "cuda"))
    cache_dir = clipcfg.get("cache_dir", "results/cache_clip")

    # ---- §2a clean-CUB features (the feature head trains on CLEAN, not composited, images) ----
    cub_root = prepare_cub(cfg["cub"]["root"], download=cfg["cub"].get("download", False),
                           url=cfg["cub"].get("url"))
    clean_feats, clean_paths = {}, {}
    for sp in ("train",) + pool_splits:
        cp = clean_cub_image_paths(bundle.paths[sp], cub_root)
        clean_paths[sp] = cp
        clean_feats[sp] = clip_image_features(cp, *cargs, cache_dir, tag=f"cleancub_{sp}")
        assert_l2_normalized(clean_feats[sp], tag=f"clean-CUB {sp} features")
        # §1.1 guard: clean-CUB encode order must still match the species labels parsed from paths
        assert_label_alignment(cp, bundle.paths[sp], tag=f"cleancub_{sp}")

    clean_eval = clean_feats.get("d_test", clean_feats[pool_splits[-1]])
    clean_eval_y = bundle.species.get("d_test", bundle.species[pool_splits[-1]])

    # ---- §1.2 KNOWN-GOOD baseline FIRST (standard standardized CLIP linear probe) ----
    base = clip_linear_probe_baseline(clean_feats["train"], bundle.species["train"],
                                      clean_eval, clean_eval_y, C=C, max_iter=max_iter, seed=seed)
    print(f"[§1.2] known-good CLIP linear-probe baseline (clean CUB-200): top-1={base['top1']:.3f} "
          f"(n_train={base['n_train']}, n_eval={base['n_eval']}, classes={base['n_classes']})")
    if not base["passes"]:
        raise FeatureHeadGateError(
            base["top1"], where="§1.2 known-good baseline",
            diagnosis=("A STANDARD standardized CLIP ViT-B/32 linear probe on CLEAN CUB-200 cannot "
                       "reach 0.55 -- the bug is in FEATURE EXTRACTION or LABEL ALIGNMENT, not the "
                       "study head. Check: CLIP preprocess parity train/test, species id parsing "
                       "(0- vs 1-based), and the clean-CUB path mapping. Do NOT proceed."))

    # ---- §2a/§1.1 study feature head (STANDARDIZED -- the head-fix) on clean CUB ----
    feat_head = fit_species_head(clean_feats["train"], bundle.species["train"], C=C,
                                 max_iter=max_iter, seed=seed)
    feat_top1_cleancub = top1(feat_head, clean_eval, clean_eval_y)
    print(f"[§2a] clean-CUB feature linear-probe top-1 = {feat_top1_cleancub:.3f} "
          f"(GATE >= {GATE_MIN_TOP1}; 0.162 in the prior broken v2 run)")

    # ---- §1.4 HARD GATE: halt before ANY 2x2/verdict if the head is sub-threshold ----
    if feat_top1_cleancub < GATE_MIN_TOP1:
        raise FeatureHeadGateError(
            feat_top1_cleancub, where="§2a study head",
            diagnosis=("Feature head below the 0.55 floor after the standardized-probe fix. The "
                       "known-good baseline passed, so feature extraction is OK -- investigate the "
                       "species-head config (C / max_iter / class coverage) before any re-run. "
                       "Per §0b: NO 2x2 and NO verdict are produced."))

    # ---- feature posteriors on the CONTAMINATED (composited) pool: the contaminated representation
    for sp in pool_splits:
        assert_l2_normalized(bundle.features[sp], tag=f"composited {sp} features")
    feat_X = np.concatenate([bundle.features[s] for s in pool_splits], axis=0)
    feat_probs = head_probs(feat_head, feat_X, n_classes).astype(np.float32)

    # ---- §2b image-derived concept posteriors (standardized heads throughout) ----
    sp_y_train = bundle.species["train"]
    if concept_source == "cbm":
        probes = fit_attribute_probe(bundle.features["train"], bundle.attrs["train"],
                                     C=C, seed=seed)
        cpt_train = predict_attribute_features(probes, bundle.features["train"])
        cpt_pool = predict_attribute_features(
            probes, np.concatenate([bundle.features[s] for s in pool_splits], axis=0))
    elif concept_source == "zeroshot":
        zs_train = clip_zeroshot_attribute_features(bundle.features["train"], bundle.attr_names, *cargs)
        cpt_train = zs_train
        cpt_pool = clip_zeroshot_attribute_features(
            np.concatenate([bundle.features[s] for s in pool_splits], axis=0),
            bundle.attr_names, *cargs)
    else:  # gt_attrs_leaky (the prior, invalid path)
        cpt_train = bundle.attrs["train"]
        cpt_pool = np.concatenate([bundle.attrs[s] for s in pool_splits], axis=0)
    cpt_head = fit_species_head(cpt_train, sp_y_train, C=C, max_iter=max_iter, seed=seed)
    cpt_probs = head_probs(cpt_head, cpt_pool, n_classes).astype(np.float32)

    # ---- assemble population (same shape as make_smoke_population) ----
    species = np.concatenate([bundle.species[s] for s in pool_splits])
    place = np.concatenate([bundle.place[s] for s in pool_splits])
    s_type = bundle.species_type[species]
    typ = typicality_group(place, s_type)
    n_atyp = int((typ == ATYPICAL).sum())
    info = dict(bundle.info)
    info.update({"pool_n": int(len(species)), "pool_n_atypical": n_atyp,
                 "pool_rho": realized_rho(typ), "concept_source": concept_source,
                 "feat_top1_cleancub": feat_top1_cleancub, "baseline_top1": base["top1"],
                 "gate_min_top1": GATE_MIN_TOP1, "gate_passed": True})
    if n_atyp < 200:
        print(f"[e1] WARNING: thin atypical pool (n_atypical={n_atyp}); worst-group estimates at "
              f"low ρ will be high-variance.")
    return {
        "species": species.astype(np.int64), "species_type": s_type.astype(np.int64),
        "place": place.astype(np.int64), "typicality": typ,
        "feat_probs": feat_probs, "cpt_probs": cpt_probs, "n_classes": n_classes,
        "feat_top1": float((feat_probs.argmax(1) == species).mean()),
        "cpt_top1": float((cpt_probs.argmax(1) == species).mean()),
        "feat_top1_cleancub": feat_top1_cleancub, "baseline_top1": base["top1"],
        "gate_passed": True, "concept_source": concept_source,
        "synthetic": False, "info": info,
    }


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
