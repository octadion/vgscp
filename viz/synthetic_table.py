"""Generate LaTeX tables + a results.json digest from the synthetic theory run (Section 12).

    python -m viz.synthetic_table --run results/runs/<run_id>

Produces:
  results/tables/synthetic_p2.tex      : per-signal minority AUROC / contamination / MI / AURC
                                         at the default shortcut strength (the P2 table)
  results/tables/synthetic_regime.tex  : V_full advantage (Delta minority-AUROC vs trust) across
                                         the shortcut-strength sweep (the regime map)
  results/results.json (merged)        : every reported synthetic number keyed by claim
"""
from __future__ import annotations

import argparse
import json
import os

PRETTY = {
    "conf_msp": "Confidence (MSP)", "trust": "Trust score",
    "ensemble_disagree": "Ensemble disagree", "mcdropout": "MC-dropout",
    "V_comp": "$V_{\\mathrm{comp}}$", "V_sound": "$V_{\\mathrm{sound}}$",
    "V_full": "$V_{\\mathrm{full}}$ (ours)",
}
ORDER = ["conf_msp", "trust", "ensemble_disagree", "V_comp", "V_sound", "V_full"]


def p2_table(res: dict) -> str:
    s = res["p2_default_strength"]
    per = res["p2_summary"]
    lines = [
        "\\begin{table}[t]\\centering",
        f"\\caption{{Synthetic testbed (P2) at shortcut strength $s={s}$. Minority "
        "error-detection AUROC (higher=useful), class-conditional contamination AUROC and MI "
        "(lower=robust), and minority selective-risk AURC (lower=better). $V$ is both robust and "
        "useful; confidence/trust are contaminated and fail on the minority.}",
        "\\label{tab:synth_p2}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Signal & min AUROC$\\uparrow$ & contam AUROC$\\downarrow$ & MI$\\downarrow$ & min AURC$\\downarrow$ \\\\",
        "\\midrule",
    ]
    for sig in ORDER:
        if sig not in per:
            continue
        p = per[sig]
        lines.append(
            f"{PRETTY.get(sig, sig)} & {p['auroc_minority_mean']:.3f} "
            f"& {p['contamination_auroc_mean']:.3f} & {p['mutual_info_mean']:.4f} "
            f"& {p['aurc_minority_mean']:.3f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def regime_table(res: dict) -> str:
    agg = res["aggregate_by_strength"]
    strengths = sorted(agg.keys(), key=float)
    lines = [
        "\\begin{table}[t]\\centering",
        "\\caption{Regime map: $V_{\\mathrm{full}}$ minority-AUROC advantage over trust and "
        "ensemble across the shortcut-strength sweep (mean over seeds). Positive = $V$ wins. "
        "The honest regime where the advantage holds vs vanishes.}",
        "\\label{tab:synth_regime}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "$s$ & minority acc & $V_{\\mathrm{full}}$ minAUROC & $\\Delta$ vs trust & $\\Delta$ vs ens. \\\\",
        "\\midrule",
    ]
    for s in strengths:
        a = agg[s]
        v = a["per_signal"]["V_full"]["auroc_minority_mean"]
        t = a["per_signal"]["trust"]["auroc_minority_mean"]
        e = a["per_signal"]["ensemble_disagree"]["auroc_minority_mean"]
        lines.append(f"{float(s):.2f} & {a['minority_acc_mean']:.3f} & {v:.3f} "
                     f"& {v - t:+.3f} & {v - e:+.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="results/tables")
    args = ap.parse_args()

    with open(os.path.join(args.run, "synthetic_results.json")) as f:
        res = json.load(f)
    os.makedirs(args.out, exist_ok=True)
    for name, tex in [("synthetic_p2", p2_table(res)), ("synthetic_regime", regime_table(res))]:
        path = os.path.join(args.out, f"{name}.tex")
        with open(path, "w") as f:
            f.write(tex)
        print(f"wrote {path}")

    # merge into results/results.json keyed by claim
    results_json_path = os.path.join("results", "results.json")
    digest = {}
    if os.path.exists(results_json_path):
        with open(results_json_path) as f:
            digest = json.load(f)
    digest["synthetic_P1_validity"] = res["p1_validity"]
    digest["synthetic_P2_default_strength"] = res["p2_default_strength"]
    digest["synthetic_P2_summary"] = res["p2_summary"]
    digest["synthetic_regime_map"] = {
        s: {"V_full_minAUROC": res["aggregate_by_strength"][s]["per_signal"]["V_full"]["auroc_minority_mean"],
            "trust_minAUROC": res["aggregate_by_strength"][s]["per_signal"]["trust"]["auroc_minority_mean"],
            "ensemble_minAUROC": res["aggregate_by_strength"][s]["per_signal"]["ensemble_disagree"]["auroc_minority_mean"],
            "minority_acc": res["aggregate_by_strength"][s]["minority_acc_mean"]}
        for s in res["aggregate_by_strength"]
    }
    with open(results_json_path, "w") as f:
        json.dump(digest, f, indent=2)
    print(f"merged synthetic numbers into {results_json_path}")


if __name__ == "__main__":
    main()
