"""Per-sample logging schema (Section 10 of the spec).

One row per (sample, split). Log EVERYTHING so any new analysis/figure is possible without
re-running. Columns whose value is a structure (per-label sets, concept lists) are stored as
JSON strings to stay parquet-friendly.
"""
from __future__ import annotations

# Canonical column order. Conformal per-score columns are templated over {THR, APS, RAPS}
# and over the base score vs verifier-aware score (sV).
IDENTITY_COLS = [
    "sample_id",
    "dataset",
    "split",
    "seed",
]

LABEL_COLS = [
    "y_true",
    "y_pred",
    "correct",
    "group_id",
    "spurious_attr",
    "is_minority",
]

PROB_COLS = [
    "p_true",       # p(y_true | x)
    "conf_msp",     # max_y p(y|x)
    "entropy",
    "margin",       # top1 - top2 probability
]

SIGNAL_COLS = [
    "trust",
    "ensemble_disagree",
    "mcdropout_var",
    "V_comp",
    "V_sound",
    "V_full",
    "R_adv",
    "reject_prob",
]

NCV_COLS = [
    "merlin_concepts",      # json
    "morgana_concepts",     # json
    "pA_pred_given_SM",
    "pA_pred_given_SA",
]

# Base nonconformity score of the TRUE label, per base score type.
SCORE_TRUE_COLS = [
    "score_THR_true",
    "score_APS_true",
    "score_RAPS_true",
]

# Conformal set membership / coverage / size, templated per base score and per score family
# (base vs verifier-aware sV). These are filled by the conformal stage given a qhat.
def conformal_cols(score_families=("THR", "APS", "RAPS"), variants=("base", "sV")):
    cols = []
    for v in variants:
        suffix = "" if v == "base" else "_sV"
        for s in score_families:
            cols += [
                f"in_set_{s}{suffix}",     # json: per-label membership 0/1
                f"set_size_{s}{suffix}",
                f"covered_{s}{suffix}",    # 1 if y_true in set
            ]
    return cols


GATE_COLS_TEMPLATE = "gate_{signal}"          # gate value per signal == the signal itself
RETAINED_COLS_TEMPLATE = "retained_{signal}_b{budget}"  # 1 if retained at budget b
QHAT_COLS_TEMPLATE = "qhat_{method}_{score}"  # filled at method level (constant per split)

MISC_COLS = [
    "wall_clock_stage",
]

SIGNAL_NAMES = [
    "conf_msp",
    "trust",
    "ensemble_disagree",   # already "higher = more reliable" (negative disagreement)
    "mcdropout",           # negative variance
    "V_comp",
    "V_sound",
    "V_full",
]


def base_columns(score_families=("THR", "APS", "RAPS")):
    """Return the full non-templated column list (templated gate/retained/qhat added later)."""
    return (
        IDENTITY_COLS
        + LABEL_COLS
        + PROB_COLS
        + SIGNAL_COLS
        + NCV_COLS
        + SCORE_TRUE_COLS
        + conformal_cols(score_families)
        + MISC_COLS
    )
