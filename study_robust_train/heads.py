"""Last-layer heads on frozen features: ERM and DFR.

Both reuse the AUDIT-VERIFIED-GOOD standardized probe from ``experiments.real_data``
(``fit_species_head`` = StandardScaler -> multinomial LogisticRegression; the v3 head-fix)
and its L2-normalization guard (``assert_l2_normalized``). See AUDIT_study.md §1.

  * ERM   : standardized logistic head fit on the composited (in-domain) train split. No
            group balancing -> it is free to lean on the spurious background shortcut.
  * DFR   : Deep Feature Reweighting (Kirichenko et al. 2022) in the frozen-feature,
            last-layer-only regime. Retrain the last layer on GROUP-BALANCED data, averaged
            over several balanced subsamples (the original averages last-layer weights over
            ~10 group-balanced subsets; we average predicted probabilities, which is
            equivalent in expectation and robust across the per-subset StandardScaler).

§2 hardening (AUDIT_study.md Gap A): ``assert_multinomial_safe`` makes the multinomial
guarantee explicit and version-safe so the standardized recipe cannot silently drift to
one-vs-rest on a future sklearn bump. On the binary Waterbirds Phase-0 task this is moot;
it matters for the multiclass species arms in the full grid.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sklearn

# Reuse the audit-verified-good standardized probe + L2 guard (no torch pulled at import).
from experiments.real_data import assert_l2_normalized, fit_species_head, head_probs, top1

__all__ = [
    "assert_multinomial_safe",
    "fit_erm",
    "fit_dfr",
    "DFRHead",
    "head_probs",
    "top1",
    "assert_l2_normalized",
]

_SKL_VER = tuple(int(x) for x in sklearn.__version__.split(".")[:2])


def assert_multinomial_safe() -> None:
    """§2 Gap-A guard: confirm the standardized recipe is GUARANTEED multinomial here.

    ``fit_species_head`` builds ``LogisticRegression`` with the default lbfgs solver and
    deliberately omits the (deprecated) ``multi_class`` kwarg. On sklearn >= 1.5 the lbfgs
    default IS multinomial and passing ``multi_class`` warns; on older sklearn the default
    was 'auto' which also resolves to multinomial for lbfgs. We assert the installed version
    is one where lbfgs => multinomial, so the recipe can't silently become one-vs-rest.
    """
    # lbfgs has defaulted to multinomial for multiclass since well before 1.0; the only risk
    # is a future major that flips the default. Fail loudly if we're on an unexpected version.
    if _SKL_VER < (1, 0):
        raise RuntimeError(
            f"[heads §2a] sklearn {sklearn.__version__} predates the lbfgs=>multinomial "
            f"guarantee; pin multi_class='multinomial' explicitly before trusting the head."
        )
    # Forward guard: if a future sklearn (>= 2.0) ships, re-verify the default is still
    # multinomial rather than assuming it. (Soft: warn via assertion message, don't crash a
    # known-good run.)
    if _SKL_VER >= (2, 0):
        raise RuntimeError(
            f"[heads §2a] sklearn {sklearn.__version__} is newer than this study validated "
            f"against (1.5.x). Re-confirm lbfgs still defaults to multinomial before running."
        )


def fit_erm(X_train: np.ndarray, y_train: np.ndarray, *, C: float = 1.0,
            max_iter: int = 5000, seed: int = 0):
    """ERM last-layer head: standardized multinomial logistic on the in-domain train split.

    No group balancing -> the head may exploit the spurious correlation present in the
    composited train distribution. Asserts L2-normalized inputs first (§2a).
    """
    assert_multinomial_safe()
    assert_l2_normalized(X_train, tag="ERM train features")
    return fit_species_head(X_train, y_train, C=C, max_iter=max_iter, seed=seed)


@dataclass
class DFRHead:
    """Group-balanced ensemble of standardized logistic heads (frozen-feature DFR).

    ``predict``/``predict_proba`` average the member posteriors. ``classes_`` is the union of
    member classes (binary Waterbirds: both present in every balanced subset).
    """
    members: list           # list of fitted sklearn Pipelines
    classes_: np.ndarray
    n_subsets: int
    subset_size_per_group: int

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Average each member's posterior, aligned to the union class order.
        acc = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float64)
        cls_index = {c: i for i, c in enumerate(self.classes_)}
        for m in self.members:
            p = m.predict_proba(X)
            for j, c in enumerate(m.classes_):
                acc[:, cls_index[c]] += p[:, j]
        return acc / len(self.members)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def fit_dfr(X_train: np.ndarray, y_train: np.ndarray, group_train: np.ndarray, *,
            n_subsets: int = 10, C: float = 1.0, max_iter: int = 5000, seed: int = 0) -> DFRHead:
    """DFR last-layer retrain: average standardized heads over group-balanced subsamples.

    Each of ``n_subsets`` draws subsamples every group down to the smallest group's count
    (group-balanced, without replacement), fits the standardized multinomial head, and the
    ensemble averages their posteriors. This is the frozen-feature, last-layer-only DFR: the
    feature extractor is fixed; only the linear head is re-fit on group-balanced data.

    NOTE (textbook DFR): the original fits the balanced retrain on a HELD-OUT reweighting
    split. Phase-0 keeps it in-domain on the composited ``train`` split (still composited /
    spurious distribution). The Colab notebook can pass ``d_learn`` here for the held-out
    variant without changing this code.
    """
    assert_multinomial_safe()
    assert_l2_normalized(X_train, tag="DFR train features")
    y = np.asarray(y_train)
    g = np.asarray(group_train)
    groups = np.unique(g)
    per_group = int(min((g == grp).sum() for grp in groups))
    if per_group == 0:
        raise ValueError("[DFR] at least one group is empty in the train split; cannot balance.")

    members = []
    classes_union: set = set()
    for s in range(n_subsets):
        rng = np.random.default_rng(seed * 10_000 + s)
        idx = np.concatenate([
            rng.choice(np.where(g == grp)[0], size=per_group, replace=False) for grp in groups
        ])
        rng.shuffle(idx)
        clf = fit_species_head(X_train[idx], y[idx], C=C, max_iter=max_iter, seed=seed * 10_000 + s)
        members.append(clf)
        classes_union.update(int(c) for c in clf.classes_)

    classes_ = np.array(sorted(classes_union))
    return DFRHead(members=members, classes_=classes_, n_subsets=n_subsets,
                   subset_size_per_group=per_group)
