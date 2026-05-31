#!/usr/bin/env bash
# Reproduce Phase-1 results end-to-end (Section 15).
# Usage: bash scripts/reproduce.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
TS="${RUN_TS:-phase1}"   # fixed timestamp -> deterministic run id (Date.now is avoided in code)

echo "==> [1/4] Unit tests (conformal quantile + gate validity + metrics)"
"$PY" -m pytest tests/ -q

echo "==> [2/4] Synthetic theory testbed (P1/P2) — CPU, no torch"
"$PY" -m theory.run_synthetic --config configs/synthetic.yaml --timestamp "$TS"

# The synthetic run already emits results/figures/theory_p2.pdf and per-sample parquet logs.

if [ "${RUN_CLEVR:-0}" = "1" ]; then
  echo "==> [3/4] CLEVR-Hans3 kill-switch (requires GPU + CLEVR_HANS3_ROOT)"
  : "${CLEVR_HANS3_ROOT:?set CLEVR_HANS3_ROOT to the extracted dataset}"
  "$PY" -m scripts.run_phase1 --config configs/clevr_hans3.yaml --timestamp "$TS"

  RUN_DIR="results/runs/clevr_hans3_phase1_${TS}"
  echo "==> [4/4] Figures + LaTeX tables from cached logs"
  "$PY" -m viz.make_figures --run "$RUN_DIR"
  "$PY" -m viz.latex_tables --run "$RUN_DIR"
else
  echo "==> [3/4] Skipping CLEVR-Hans3 (set RUN_CLEVR=1 with a GPU + CLEVR_HANS3_ROOT)."
  echo "==> [4/4] Theory figure at results/figures/theory_p2.pdf"
fi

echo "Done. Review PHASE1_REPORT.md for the GO/NO-GO verdict."
