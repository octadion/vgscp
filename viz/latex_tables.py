"""Auto-generate paper-ready LaTeX tables (booktabs) from per-sample logs (Section 12).

    python -m viz.latex_tables --run results/runs/<run_id>

Tables:
  - main_results.tex      : error-detection AUROC (overall / majority / minority) per signal,
                            value +/- 95% CI, best per column bolded; minority AURC.
  - contamination.tex     : contamination AUROC + conditional MI per signal (P2).
Values come from eval/phase1_eval on the cached TEST rows, so tables and figures are consistent.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from eval.phase1_eval import evaluate_signals

SIGNALS = ["conf_msp", "trust", "ensemble_disagree", "mcdropout", "V_comp", "V_sound", "V_full"]
PRETTY = {
    "conf_msp": "Confidence (MSP)", "trust": "Trust score",
    "ensemble_disagree": "Ensemble disagree", "mcdropout": "MC-dropout",
    "V_comp": "$V_{\\mathrm{comp}}$", "V_sound": "$V_{\\mathrm{sound}}$",
    "V_full": "$V_{\\mathrm{full}}$ (ours)",
}


def _fmt_ci(ci, best=False):
    if ci is None or not np.isfinite(ci.estimate):
        return "--"
    s = f"{ci.estimate:.3f}\\,\\tiny[{ci.lo:.3f}, {ci.hi:.3f}]"
    return f"\\textbf{{{s}}}" if best else s


def _best_idx(values, higher_better=True):
    vals = [v if v is not None and np.isfinite(v) else (-np.inf if higher_better else np.inf)
            for v in values]
    return int(np.argmax(vals) if higher_better else np.argmin(vals))


def main_results_table(reports) -> str:
    sigs = [s for s in SIGNALS if s in reports]
    cols = ["auroc_overall", "auroc_majority", "auroc_minority"]
    best = {c: _best_idx([getattr(reports[s], c).estimate if getattr(reports[s], c) else None
                          for s in sigs], True) for c in cols}
    aurc_vals = [reports[s].aurc_minority.estimate if reports[s].aurc_minority else None for s in sigs]
    best_aurc = _best_idx(aurc_vals, higher_better=False)

    lines = [
        "\\begin{table}[t]\\centering",
        "\\caption{Phase-1 error-detection AUROC (higher better) and minority selective-risk "
        "AURC (lower better). Values are mean with 95\\% bootstrap CI; best per column in bold. "
        "The kill-switch compares $V_{\\mathrm{full}}$ against trust and ensemble on the minority.}",
        "\\label{tab:main}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Signal & AUROC (overall) & AUROC (majority) & AUROC (minority) & AURC (minority)$\\downarrow$ \\\\",
        "\\midrule",
    ]
    for i, s in enumerate(sigs):
        r = reports[s]
        row = (
            f"{PRETTY.get(s, s)} & {_fmt_ci(r.auroc_overall, i == best['auroc_overall'])} "
            f"& {_fmt_ci(r.auroc_majority, i == best['auroc_majority'])} "
            f"& {_fmt_ci(r.auroc_minority, i == best['auroc_minority'])} "
            f"& {_fmt_ci(r.aurc_minority, i == best_aurc)} \\\\"
        )
        lines.append(row)
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def contamination_table(reports) -> str:
    sigs = [s for s in SIGNALS if s in reports]
    lines = [
        "\\begin{table}[t]\\centering",
        "\\caption{P2 contamination: dependence of each signal on the spurious attribute "
        "(class-conditional). Lower = more robust. $V$ should be near 0.5; confidence/trust "
        "higher.}",
        "\\label{tab:contam}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Signal & Contamination AUROC$\\downarrow$ & MI (cond.)$\\downarrow$ \\\\",
        "\\midrule",
    ]
    for s in sigs:
        r = reports[s]
        lines.append(f"{PRETTY.get(s, s)} & {r.contamination_auroc:.3f} & {r.mutual_info:.4f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="results/tables")
    args = ap.parse_args()

    from vgscp_logging.parquet_logger import load_run_logs

    df = load_run_logs(args.run, split="test")
    reports = evaluate_signals(df, [s for s in SIGNALS if s in df.columns])
    os.makedirs(args.out, exist_ok=True)
    for name, tex in [("main_results", main_results_table(reports)),
                      ("contamination", contamination_table(reports))]:
        path = os.path.join(args.out, f"{name}.tex")
        with open(path, "w") as f:
            f.write(tex)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
