# Paste this as a NEW CELL at the end of the Colab session that just failed, then run it.
# Do not restart the runtime: `pg` and `sup` are still in memory, so the 40-minute loop does not
# repeat and this takes seconds.
#
# The in-run check already passed perfectly -- 2400 rows, max coverage difference 0.00e+00, zero
# argmin mismatches -- so the recomputed per-group minimum equals evaluate()'s own worst_group_cov
# everywhere and the fresh run is internally sound. What disagrees is the RELEASED CSV, by up to
# 1.29e-02 per row, while the per-setting means still match to under 6e-4. Per-row differences that
# cancel in the mean point at either a cold feature cache (ResNet-50 is trained in-domain, so its
# features are stochastic) or a changed head fit, and the pattern below separates the two.
#
# It writes the CSV first, so the run cannot be lost to a failed check a second time.

import numpy as np
import pandas as pd

pg.to_csv(OUT_PERGROUP, index=False)
print("saved ->", OUT_PERGROUP, pg.shape)

pub = pd.read_csv(PUB_CSV)
pub = pub[(pub.score == SCORE) & (pub.rho_test == RHO)]
key = ["backbone", "dataset", "method", "train_seed", "calibration", "split_seed"]
print(f"duplicate keys in the released CSV: {pub.duplicated(key).sum()}")

# `pg` names its published copies worst_group_pub / worst_group_cov_pub, so nothing collides on the
# merge and pandas applies no suffix -- the released columns arrive under their own names.
mg = pg.merge(pub[key + ["worst_group_cov", "worst_group", "gate_status"]], on=key, how="inner")
mg["d"] = (mg["cov_min_recomputed"] - mg["worst_group_cov"]).abs()
mg["argmin_differs"] = mg["argmin_recomputed"] != mg["worst_group"]

print(f"\nrows matched   : {len(mg)}")
print(f"rows differing : {(mg.d > 1e-9).sum()}  ({100 * (mg.d > 1e-9).mean():.1f}%)")
print(f"max difference : {mg.d.max():.2e}")
print(f"argmin differs : {int(mg.argmin_differs.sum())}")

print("\n=== where the disagreement sits ===")
for by in (["backbone"], ["dataset"], ["method"], ["calibration"], ["train_seed"],
           ["backbone", "method"]):
    t = mg.groupby(by)["d"].agg(n="size", frac_diff=lambda s: (s > 1e-9).mean(), worst="max")
    t = t[t.worst > 1e-9].sort_values("worst", ascending=False)
    if len(t):
        print(f"\n-- by {'+'.join(by)}")
        print(t.head(12).round(5).to_string())

clean = mg[mg.d <= 1e-9]
print(f"\n=== reproduces exactly ({len(clean)} of {len(mg)} rows) ===")
if len(clean):
    print(clean.groupby(["backbone", "method"]).size().to_string())

print("\n=== ten worst rows ===")
print(mg.nlargest(10, "d")[key + ["cov_min_recomputed", "worst_group_cov", "d", "gate_status"]]
        .round(5).to_string(index=False))

# The sharpest probe available: ERM and AFR fit with L-BFGS, whose solver ignores the seed, so a
# rerun must reproduce them exactly unless the FEATURES changed underneath.
det = mg[mg.method.isin(["erm", "afr"])]
sto = mg[~mg.method.isin(["erm", "afr"])]
print(f"\ndeterministic heads (erm, afr) : {len(det)} rows, "
      f"{(det.d > 1e-9).sum()} differing, max {det.d.max():.2e}")
print(f"stochastic heads               : {len(sto)} rows, "
      f"{(sto.d > 1e-9).sum()} differing, max {sto.d.max():.2e}")
print("\nIf the deterministic heads move too, the features differ between the two runs "
      "(a cold cache would do it) and head fitting is not the cause.")

# The per-group answer the paper needs, printed here so it survives even if the download fails.
cols = [f"cov_g{g}" for g in range(4)]
g = pg.groupby(["dataset", "backbone", "calibration"])
res = pd.DataFrame({"mean_over_groups": g[cols].mean().mean(axis=1),
                    "mean_of_minima": g["cov_min_recomputed"].mean()})
res["gap"] = res["mean_over_groups"] - res["mean_of_minima"]
print("\n=== the min-over-groups effect, measured rather than simulated ===")
print(res.round(4).to_string())

from google.colab import files
files.download(OUT_PERGROUP)
