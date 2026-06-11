"""study_robust_train — Pre-Registered Campaign v2 ("Conformal Burden").

Clean module for the v2 hardened study. Imports ONLY the audit-verified-live pieces
(see AUDIT_study.md §5): conformal.{scores,split_conformal,group_robust},
experiments.real_data head helpers, experiments.shift_resampler. No legacy v1-v4
cluster (signals/registry, models/verifier*, ks_conformal, legacy verdicts) is reachable
from here.

THIS TURN: Phase-0 only (ERM + DFR last-layer on Waterbirds, frozen features). The
full grid (H1/H2/H3) is NOT implemented here by design — Phase-0-then-STOP holds.
"""
