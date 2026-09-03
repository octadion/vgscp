"""Is run_grid_streaming genuinely equivalent to run_grid, or just a way to dodge the crash?

    python -m study_robust_train.validate_grid_streaming

Four things must hold, and the third and fourth were untested:
  1. identical records and verdicts on a single pass
  2. the gate's semantics survive: excluded/flagged arms and verdicts_with_excluded
  3. PARTIAL completion then resume equals one uninterrupted run
  4. CSV round-trip does not lose float precision (resumed records come from text)
"""
import os, sys, tempfile
sys.path.insert(0, r"C:\jagr\vgscp")
import numpy as np
from study_robust_train import grid as G
from study_robust_train.grid import (records_from_csv, run_grid, run_grid_streaming,
                                     write_csv)
from study_robust_train.synthetic import make_synthetic_griddata

FAIL = []
def chk(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    if not cond: FAIL.append(name)

KEYS = [("bbA", "waterbirds"), ("bbB", "waterbirds"), ("bbA", "celeba"), ("bbB", "celeba")]
SPUR = {"bbA": 1.9, "bbB": 0.8}
# Fixed, not hash(): Python salts string hashes per process, so hash() made this test
# non-reproducible -- the synthetic data differed between runs and the gate outcome moved with it.
SEED_OF = {("bbA", "waterbirds"): 11, ("bbB", "waterbirds"): 22,
           ("bbA", "celeba"): 33,     ("bbB", "celeba"): 44}
def build(bb, ds):
    return make_synthetic_griddata(backbone=bb, dataset=ds, n_pool=2400, d=16,
                                   core_scale=1.2, spur_scale=SPUR[bb],
                                   seed=SEED_OF[(bb, ds)])

KW = dict(methods=("erm", "dfr", "groupdro_ll"), scores=("APS",), rho_sweep=(0.95, 0.5),
          seeds=(0, 1), n_splits=3)
ID = ("backbone", "dataset", "method", "train_seed", "score", "rho_test", "split_seed",
      "gate_status", "calibration")
NUM = ("worst_group_cov", "mean_group_cov", "cov_range", "cov_gap", "worst_group_acc",
       "mean_set_size", "div_wasserstein1", "gate_floor")

def sig(recs, prec=None):
    out = []
    for r in recs:
        k = tuple(r[c] for c in ID)
        v = tuple((r[c] if prec is None else round(r[c], prec)) for c in NUM)
        out.append(k + v)
    return sorted(out)

# ---- force the gate to actually exclude something, so its machinery is exercised
G.WG_ACC_FLOOR = {"waterbirds": {"erm": 0.99, "dfr": 0.0, "groupdro_ll": 0.0},
                  "celeba":     {"erm": 0.0,  "dfr": 0.0, "groupdro_ll": 0.99}}
# 1.01 so the soft band is certain to trigger: the point is to exercise the code path,
# and at 0.99 DFR on this synthetic pool sat above the threshold and never flagged.
G.WG_ACC_SOFT_FLAG = {("celeba", "dfr"): 1.01}

print("\n[1] single pass: streaming vs batched")
batched = run_grid({k: build(*k) for k in KEYS}, **KW)
td = tempfile.mkdtemp()
csv1 = os.path.join(td, "one.csv")
streamed = run_grid_streaming(KEYS, build, cell_csv=csv1, verbose=False, **KW)
chk("record count equal", len(batched["records"]) == len(streamed["records"]),
    f"{len(batched['records'])} vs {len(streamed['records'])}")
chk("records identical to full float precision", sig(batched["records"]) == sig(streamed["records"]))
chk("verdicts identical", batched["verdicts"] == streamed["verdicts"])

print("\n[2] the gate's semantics survive")
def norm(lst):
    return sorted((d["backbone"], d["dataset"], d["method"], d["seed"]) for d in lst)
chk("gate actually excluded something", len(batched["excluded"]) > 0,
    f"{len(batched['excluded'])} arm(s)")
chk("excluded list identical", norm(batched["excluded"]) == norm(streamed["excluded"]))
chk("flagged list identical", norm(batched["flagged"]) == norm(streamed["flagged"]))
chk("verdicts_with_excluded present in both",
    ("verdicts_with_excluded" in batched) and ("verdicts_with_excluded" in streamed))
chk("sensitivity verdicts identical",
    batched.get("verdicts_with_excluded") == streamed.get("verdicts_with_excluded"))
st = {}
for r in streamed["records"]: st[r["gate_status"]] = st.get(r["gate_status"], 0) + 1
chk("all three gate_status values appear", set(st) == {"kept", "flagged", "excluded"}, str(st))

print("\n[3] partial completion then resume == one uninterrupted run")
csv2 = os.path.join(td, "partial.csv")
part = run_grid_streaming(KEYS[:2], build, cell_csv=csv2, verbose=False, **KW)   # first 2 cells
chk("partial pass did 2 cells",
    len({(r["backbone"], r["dataset"]) for r in part["records"]}) == 2)
resumed = run_grid_streaming(KEYS, build, cell_csv=csv2, verbose=False, **KW)    # resume all 4
chk("resumed covers all 4 cells",
    len({(r["backbone"], r["dataset"]) for r in resumed["records"]}) == 4)
chk("resumed records equal a single full run (6 dp)",
    sig(resumed["records"], 6) == sig(batched["records"], 6))
chk("resumed verdicts equal a single full run", resumed["verdicts"] == batched["verdicts"])
chk("resumed excluded list complete", norm(resumed["excluded"]) == norm(batched["excluded"]),
    f"{len(resumed['excluded'])} vs {len(batched['excluded'])}")
chk("resumed flagged list complete", norm(resumed["flagged"]) == norm(batched["flagged"]),
    f"{len(resumed['flagged'])} vs {len(batched['flagged'])}")
chk("resumed sensitivity verdicts present", "verdicts_with_excluded" in resumed)

print("\n[4] CSV round-trip keeps float precision")
csv3 = os.path.join(td, "rt.csv")
write_csv(batched["records"], csv3)
back = records_from_csv(csv3)
chk("row count preserved", len(back) == len(batched["records"]))
chk("exact to full precision", sig(back) == sig(batched["records"]))
worst = max(abs(a - b) for ra, rb in zip(sig(back), sig(batched["records"]))
            for a, b in zip(ra[len(ID):], rb[len(ID):]))
chk("max numeric drift is zero", worst == 0.0, f"{worst:.2e}")

print("\n" + "=" * 70)
print(f"FAILED ({len(FAIL)}): " + ", ".join(FAIL) if FAIL else "STREAMING IS EQUIVALENT")
sys.exit(1 if FAIL else 0)
