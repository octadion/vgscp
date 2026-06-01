"""Unit tests for the pre-committed premise-2 verdict logic (eval.phase1_eval.premise2_verdict).

The smoke run only exercises the PARTIAL branch, so these hand-craft minority-group signal columns
that drive each branch (NOVELTY-VALIDATED / PARTIAL / NULL) deterministically. No torch needed."""
import numpy as np
import pandas as pd

from eval.phase1_eval import premise2_verdict


def _df(correct, signals, n_majority=40):
    """Build a test frame: a MINORITY block (where the verdict is evaluated) carrying the given
    per-signal arrays, plus a benign majority block (ignored by the minority-only verdict)."""
    n = len(correct)
    rows = {"correct": list(correct) + [1] * n_majority,
            "is_minority": [1] * n + [0] * n_majority}
    rng = np.random.default_rng(0)
    for name, vals in signals.items():
        rows[name] = list(vals) + list(rng.random(n_majority))
    return pd.DataFrame(rows)


def _separating(correct, rng, strength):
    """A signal that ranks correct>incorrect with controllable separation (higher=more reliable)."""
    correct = np.asarray(correct)
    return correct * strength + rng.normal(0, 1.0, len(correct))


def test_novelty_validated():
    rng = np.random.default_rng(1)
    n = 200
    correct = rng.integers(0, 2, n)
    # V_full separates strongly; every control/baseline AND V_comp separate weakly -> V wins all.
    sig = {
        "V_full": _separating(correct, rng, 6.0),
        "V_comp": _separating(correct, rng, 1.0),
        "probe_concept": _separating(correct, rng, 0.5),
        "trust_concept": _separating(correct, rng, 0.5),
        "trust": _separating(correct, rng, 0.5),
        "ensemble_disagree": _separating(correct, rng, 0.5),
    }
    v = premise2_verdict(_df(correct, sig), space="mixed", n_resamples=400, seed=0)
    assert v.label == "NOVELTY-VALIDATED", v.rationale
    assert v.morgana_adds_value


def test_partial_clean_concepts_help():
    rng = np.random.default_rng(2)
    n = 200
    correct = rng.integers(0, 2, n)
    # V beats f-baselines, but trust_concept is just as good (a control ties) -> PARTIAL.
    strong = _separating(correct, rng, 6.0)
    sig = {
        "V_full": strong,
        "V_comp": _separating(correct, rng, 1.0),
        "probe_concept": _separating(correct, rng, 0.4),
        "trust_concept": strong.copy(),                     # control ties V_full exactly
        "trust": _separating(correct, rng, 0.4),
        "ensemble_disagree": _separating(correct, rng, 0.4),
    }
    v = premise2_verdict(_df(correct, sig), space="mixed", n_resamples=400, seed=0)
    assert v.label == "PARTIAL", v.rationale


def test_partial_morgana_idle():
    rng = np.random.default_rng(3)
    n = 200
    correct = rng.integers(0, 2, n)
    strong = _separating(correct, rng, 6.0)
    # V_full beats ALL controls + baselines, but V_comp == V_full -> Morgana adds nothing.
    sig = {
        "V_full": strong,
        "V_comp": strong.copy(),
        "probe_concept": _separating(correct, rng, 0.4),
        "trust_concept": _separating(correct, rng, 0.4),
        "trust": _separating(correct, rng, 0.4),
        "ensemble_disagree": _separating(correct, rng, 0.4),
    }
    v = premise2_verdict(_df(correct, sig), space="mixed", n_resamples=400, seed=0)
    assert v.label == "PARTIAL", v.rationale
    assert not v.morgana_adds_value


def test_null_no_control_beaten():
    rng = np.random.default_rng(4)
    n = 200
    correct = rng.integers(0, 2, n)
    weak = _separating(correct, rng, 1.0)
    # The controls separate AT LEAST as well as V_full -> V beats no control -> NULL.
    sig = {
        "V_full": weak,
        "V_comp": weak.copy(),
        "probe_concept": _separating(correct, rng, 6.0),
        "trust_concept": _separating(correct, rng, 6.0),
        "trust": _separating(correct, rng, 6.0),
        "ensemble_disagree": _separating(correct, rng, 6.0),
    }
    v = premise2_verdict(_df(correct, sig), space="mixed", n_resamples=400, seed=0)
    assert v.label == "NULL", v.rationale
