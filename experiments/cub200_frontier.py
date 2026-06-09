"""E1 — CUB-200 multiclass coverage--efficiency frontier construction (the positive anchor).

This is the NEW multiclass infrastructure (never built before). It follows the spurious-correlation
construction & sweep specified in the run spec §2b EXACTLY; do not improvise a different spurious
axis. The construction reuses the standard Waterbirds machinery already in the repo and layers a
200-way species target on top of it:

  * TARGET (predicted label):  the fine-grained CUB-200 species id (0..199) parsed from each
    Waterbirds image's CUB folder name (e.g. ``001.Black_footed_Albatross`` -> class 1). 200 classes
    give a non-degenerate set-size axis (unlike binary Waterbirds).
  * SPURIOUS attribute:        binary background type ``place`` in {land=0, water=1}.
  * TYPICAL background:        each species' type t(species) is the Waterbirds binary label
    ``y`` (waterbird=1 / landbird=0), which is CONSTANT per species. A waterbird's typical
    background is water; a landbird's is land. So typical background of species s == t(s).
  * GROUP (for worst-group metrics): binary background-TYPICALITY
        typicality = 1 if (place == t(species))  [TYPICAL]   else 0 [ATYPICAL = minority / worst].
    This is exactly Waterbirds' concordant/minority axis (place == y), now under a 200-way target.
    WORST-GROUP COVERAGE = coverage on the ATYPICAL group (typicality == 0).
  * CORRELATION STRENGTH rho:  rho = P(place == typical) = fraction of images on the typical
    background. Calibrate at high rho (0.95); sweep test rho in {0.5,0.6,0.7,0.8,0.9,0.95} by
    resampling. Class balance (the 200-way species marginal) is held FIXED across rho so ONLY the
    spurious correlation changes -- ``resample_to_rho_multiclass`` enforces this by fixing the
    per-species draw count and only re-mixing typical/atypical within each species at ratio
    rho:(1-rho).

Two representations feed the SAME conformal score functions (APS/RAPS/THR) -- only the (N, C) class
posteriors differ (see ``signals.conformal_scores`` reused downstream):
  * FEATURE space:  a 200-way head on FROZEN CLIP global image features. Background-dominated /
    spurious-sensitive (the contaminated representation).
  * CONCEPT space:  a 200-way head on the 312 CUB per-image attributes (the shortcut-invariant
    representation; per-image MTurk attributes describe the BIRD, not the pasted background).

REAL path (Colab/GPU): ``load_real_population`` documents and wires the exact recipe (encode CLIP
features once + cache, fit two logistic heads on frozen features/attributes). It needs torch +
open_clip + the CUB/Waterbirds datasets, which are unavailable locally, so it raises a clear BLOCKED
error rather than fabricating. ``make_smoke_population`` fabricates a controlled multiclass testbed
that exercises the full construct->resample->calibrate->score->frontier->verdict pipeline on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Group convention (binary background-typicality)
ATYPICAL = 0   # minority / worst group: bird on its NON-typical background
TYPICAL = 1


# ======================================================================================
# Construction helpers (species type, typicality group, correlation strength)
# ======================================================================================
def typicality_group(place: np.ndarray, species_type: np.ndarray) -> np.ndarray:
    """Binary background-typicality group: 1 (TYPICAL) where place == species_type else 0 (ATYPICAL).

    ``species_type`` is t(species) in {0=land,1=water}; in Waterbirds this equals the binary label
    ``y`` and is constant per species. The ATYPICAL group is the minority / worst group.
    """
    place = np.asarray(place).astype(int)
    species_type = np.asarray(species_type).astype(int)
    return (place == species_type).astype(np.int64)


def realized_rho(typicality: np.ndarray) -> float:
    """Empirical correlation strength rho = P(place == typical) = fraction in the TYPICAL group."""
    t = np.asarray(typicality)
    return float((t == TYPICAL).mean()) if t.size else float("nan")


# ======================================================================================
# Multiclass correlation-strength resampler (holds the 200-way class balance FIXED)
# ======================================================================================
@dataclass
class ResampleResult:
    idx: np.ndarray            # indices INTO the pool that was passed in (rows may repeat)
    rho_target: float
    rho_realized: float
    n_typical: int
    n_atypical: int
    species_counts: dict       # species -> draw count (held FIXED across rho)
    n_empty_atypical: int      # species needing atypical draws but with none in pool (logged)


def reference_species_counts(species_pool: np.ndarray, n: int) -> dict:
    """Per-species draw counts summing to ~``n``, proportional to the pool species frequency.

    Computed ONCE and reused for every rho so the 200-way class marginal is held fixed across the
    sweep (only the typical/atypical mix within each species changes with rho). Largest-remainder
    rounding to hit the total exactly.
    """
    species_pool = np.asarray(species_pool)
    classes, freq = np.unique(species_pool, return_counts=True)
    raw = freq / freq.sum() * n
    counts = np.floor(raw).astype(int)
    rem = n - int(counts.sum())
    if rem > 0:
        order = np.argsort(-(raw - counts))
        for k in range(rem):
            counts[order[k % len(counts)]] += 1
    return {int(c): int(n) for c, n in zip(classes, counts)}


def resample_to_rho_multiclass(
    species_pool: np.ndarray,
    typicality_pool: np.ndarray,
    rho: float,
    n: int,
    seed: int,
    species_counts: Optional[dict] = None,
    replace: bool = True,
) -> ResampleResult:
    """Resample pool indices to correlation strength ``rho`` at a FIXED 200-way class balance.

    For each species the draw count is fixed (``species_counts``, computed once); within the species
    we draw ``round(count*rho)`` from its TYPICAL members and the rest from its ATYPICAL members
    (with replacement by default, so extreme rho stays reachable from a finite pool). This changes
    ONLY the spurious correlation, not the class marginal. If a species needs atypical draws but has
    no atypical members in the pool (possible for real CUB species with few minority images), those
    slots are redirected to its typical members and counted in ``n_empty_atypical`` (logged, not
    silently dropped). Returns indices INTO the passed pool.
    """
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"rho must be in [0,1], got {rho}")
    rng = np.random.default_rng(seed)
    species_pool = np.asarray(species_pool)
    typicality_pool = np.asarray(typicality_pool)
    if species_counts is None:
        species_counts = reference_species_counts(species_pool, n)

    chosen, n_typ, n_atyp, n_empty = [], 0, 0, 0
    for c, want in species_counts.items():
        if want <= 0:
            continue
        k_typ = int(round(want * rho))
        k_atyp = want - k_typ
        typ_members = np.where((species_pool == c) & (typicality_pool == TYPICAL))[0]
        atyp_members = np.where((species_pool == c) & (typicality_pool == ATYPICAL))[0]
        # redirect impossible draws (empty cell) to the other cell so the class count is preserved
        if k_atyp > 0 and atyp_members.size == 0:
            n_empty += 1
            k_typ, k_atyp = k_typ + k_atyp, 0
        if k_typ > 0 and typ_members.size == 0:
            k_atyp, k_typ = k_atyp + k_typ, 0
        if k_typ > 0:
            if (not replace) and k_typ > typ_members.size:
                raise ValueError(f"species {c}: need {k_typ} typical but only {typ_members.size}")
            chosen.append(rng.choice(typ_members, size=k_typ, replace=replace))
            n_typ += k_typ
        if k_atyp > 0:
            if (not replace) and k_atyp > atyp_members.size:
                raise ValueError(f"species {c}: need {k_atyp} atypical but only {atyp_members.size}")
            chosen.append(rng.choice(atyp_members, size=k_atyp, replace=replace))
            n_atyp += k_atyp
    if not chosen:
        raise ValueError("resample produced no samples; pool is empty or species_counts all zero")
    idx = np.concatenate(chosen)
    rng.shuffle(idx)
    return ResampleResult(
        idx=idx, rho_target=float(rho), rho_realized=realized_rho(typicality_pool[idx]),
        n_typical=int(n_typ), n_atypical=int(n_atyp), species_counts=species_counts,
        n_empty_atypical=int(n_empty),
    )


def split_pool(n_total: int, frac_cal: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Seeded disjoint split of pooled indices into (cal_pool, test_pool)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    n_cal = int(round(frac_cal * n_total))
    return np.sort(perm[:n_cal]), np.sort(perm[n_cal:])


# ======================================================================================
# Synthetic multiclass population (CPU pipeline self-test -- NOT a scientific claim)
# ======================================================================================
# Concept-source presets (run spec v2 §2b). The corrected run must NOT score ground-truth MTurk
# attributes at test time (label leakage); test-time concept scores must be IMAGE-DERIVED. We model
# three sources so the synthetic harness exercises each. Each preset = (margin, residual_delta,
# spurious_kappa): higher residual_delta => more residual non-invariance; spurious_kappa>0 => the
# concept score inherits SOME background contamination (predicted concepts read a contaminated CLIP
# feature, unlike the clean GT annotations).
CONCEPT_SOURCES = {
    # LEAKY baseline: ground-truth per-image attributes (NOT used at test in the corrected run; kept
    # only to reproduce the prior, invalid behaviour). Clean + strongly invariant + high accuracy.
    "gt_attrs_leaky": dict(cpt_margin=2.6, cpt_residual_delta=0.35, cpt_spurious_kappa=0.0),
    # PRIMARY: CBM probe (CLIP features -> predicted attributes -> species). Image-derived, so it
    # inherits a little background contamination and loses some accuracy vs the GT annotations.
    "cbm": dict(cpt_margin=2.1, cpt_residual_delta=0.55, cpt_spurious_kappa=0.35),
    # APPENDIX: CLIP zero-shot attribute cosines. Fully image-derived, no attribute training; weakest
    # signal (lowest accuracy), slightly more residual + contamination.
    "zeroshot": dict(cpt_margin=1.6, cpt_residual_delta=0.75, cpt_spurious_kappa=0.45),
}


@dataclass
class SmokeConfig:
    n: int = 9000
    n_classes: int = 40        # << 200 for a fast smoke; pipeline is class-count agnostic
    p_typical_pool: float = 0.80   # base typicality rate of the generated pool
    # FEATURE head (CLIP-like, background-contaminated): true-class margin DROPS on atypical, and a
    # spurious boost pushes mass toward species whose TYPE matches the background.
    feat_margin_typical: float = 3.4
    feat_margin_atypical: float = 1.2
    feat_spurious_kappa: float = 1.8
    # CONCEPT head: the corrected run uses an IMAGE-DERIVED (predicted) concept source -- default the
    # CBM preset (§2b). Set ``concept_source`` to pick a preset, or override the three knobs directly.
    concept_source: str = "cbm"
    cpt_margin: Optional[float] = None
    cpt_residual_delta: Optional[float] = None
    cpt_spurious_kappa: Optional[float] = None
    margin_sd: float = 0.8
    seed: int = 0

    def resolved_concept(self) -> dict:
        """Concept-head knobs: explicit overrides win, else the ``concept_source`` preset."""
        if self.concept_source not in CONCEPT_SOURCES:
            raise ValueError(f"unknown concept_source {self.concept_source!r}; "
                             f"choose from {list(CONCEPT_SOURCES)}")
        base = dict(CONCEPT_SOURCES[self.concept_source])
        if self.cpt_margin is not None:
            base["cpt_margin"] = self.cpt_margin
        if self.cpt_residual_delta is not None:
            base["cpt_residual_delta"] = self.cpt_residual_delta
        if self.cpt_spurious_kappa is not None:
            base["cpt_spurious_kappa"] = self.cpt_spurious_kappa
        return base


def _softmax_probs(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    return p.astype(np.float32)


def make_smoke_population(cfg: SmokeConfig) -> dict:
    """Fabricate a controlled CUB-200-like multiclass population for the CPU pipeline self-test.

    Returns per-sample arrays + the two (N, C) class-posterior matrices (feature head & concept
    head). The feature head is spurious-sensitive (its true-class score depends on background
    typicality), the concept head is approximately shortcut-invariant with a small residual gap.
    """
    rng = np.random.default_rng(cfg.seed)
    n, C = cfg.n, cfg.n_classes
    # fixed per-species type t(species) in {0=land,1=water}; balanced across the 200-way space
    species_type = (np.arange(C) % 2).astype(np.int64)
    species = rng.integers(0, C, n)
    s_type = species_type[species]
    # background: typical with prob p_typical_pool (independent of class -> class balance preserved)
    is_typ = rng.random(n) < cfg.p_typical_pool
    place = np.where(is_typ, s_type, 1 - s_type).astype(np.int64)
    typ = typicality_group(place, s_type)

    # ----- FEATURE head logits (contaminated) -----
    feat_logits = rng.normal(0.0, 1.0, size=(n, C))
    margin_f = np.where(typ == TYPICAL, cfg.feat_margin_typical, cfg.feat_margin_atypical)
    margin_f = np.maximum(0.05, rng.normal(margin_f, cfg.margin_sd))
    feat_logits[np.arange(n), species] += margin_f
    # spurious boost: every class whose TYPE matches the observed background gets +kappa. On atypical
    # samples this boosts the WRONG-type classes (background disagrees with the true species type),
    # inflating the true-label nonconformity score on the atypical (worst) group.
    type_match = (species_type[None, :] == place[:, None])  # (n, C)
    feat_logits += cfg.feat_spurious_kappa * type_match
    feat_probs = _softmax_probs(feat_logits)

    # ----- CONCEPT head logits (image-derived predicted concept; mostly shortcut-invariant) -----
    ck = cfg.resolved_concept()
    cpt_logits = rng.normal(0.0, 1.0, size=(n, C))
    margin_c = np.where(typ == TYPICAL, ck["cpt_margin"], ck["cpt_margin"] - ck["cpt_residual_delta"])
    margin_c = np.maximum(0.05, rng.normal(margin_c, cfg.margin_sd))
    cpt_logits[np.arange(n), species] += margin_c
    # predicted concepts read a CONTAMINATED CLIP feature, so (unlike clean GT annotations) they
    # inherit a SMALL background-spurious boost -- 0 for the gt_attrs_leaky preset.
    if ck["cpt_spurious_kappa"] > 0:
        cpt_logits += ck["cpt_spurious_kappa"] * (species_type[None, :] == place[:, None])
    cpt_probs = _softmax_probs(cpt_logits)

    return {
        "species": species.astype(np.int64),
        "species_type": s_type.astype(np.int64),
        "place": place,
        "typicality": typ,
        "feat_probs": feat_probs,
        "cpt_probs": cpt_probs,
        "n_classes": C,
        "feat_top1": float((feat_probs.argmax(1) == species).mean()),
        "cpt_top1": float((cpt_probs.argmax(1) == species).mean()),
        "concept_source": cfg.concept_source,
        "synthetic": True,
    }


# ======================================================================================
# REAL CUB-200 population (Colab/GPU path -- documented recipe; raises BLOCKED locally)
# ======================================================================================
def species_from_waterbirds_paths(paths: list[str]) -> np.ndarray:
    """Parse the 0-based CUB-200 species id from each Waterbirds image path.

    Waterbirds image paths embed the CUB folder name, e.g.
    ``.../001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg`` -> class 1 (1-based) ->
    species id 0 (0-based). Returns an (N,) int array of species ids in 0..199.
    """
    out = []
    for p in paths:
        parts = p.replace("\\", "/").rstrip("/").split("/")
        folder = parts[-2] if len(parts) >= 2 else parts[-1]
        prefix = folder.split(".", 1)[0]
        out.append(int(prefix) - 1)  # 1-based CUB class -> 0-based species id
    return np.asarray(out, dtype=np.int64)


def load_real_population(cfg: dict, seed: int) -> dict:
    """REAL CUB-200 multiclass population (CORRECTED, run spec v2 §2a + §2b). Colab/GPU only.

    Delegates to ``experiments.real_data.assemble_e1_population``, which fixes the prior confounds:
      * §2a  the FEATURE head is trained on CLEAN CUB-200 features (not background-composited ones),
        with L2-normalization asserted, and its clean top-1 reported as the sanity target (>=0.55);
        the prior run's broken 0.182 head trained on composited images. The frontier still scores the
        CONTAMINATED (Waterbirds-composited) pool posteriors -- contamination shows up in the score
        under shift, not in a broken head.
      * §2b  the CONCEPT score is IMAGE-DERIVED (predicted): cbm (default, CLIP feats->attr probe->
        predicted attrs->species), zeroshot (CLIP zero-shot attr cosines->species), or
        gt_attrs_leaky (the PRIOR INVALID path that scored ground-truth attributes at test -- kept
        ONLY for the leakage demo). Choose via ``cfg['concept_source']`` (default 'cbm').

    Returns the same dict shape as ``make_smoke_population`` (+ ``feat_top1_cleancub``,
    ``concept_source``). Requires torch + open_clip + the CUB-200 / Waterbirds image data; the
    cached-feature run is well under the 5h budget (the only training cost is the logistic heads +
    the per-attribute CBM probe). Raises clearly (never fabricates) if the data/CLIP are missing.
    """
    from experiments.real_data import assemble_e1_population
    return assemble_e1_population(cfg, seed)
