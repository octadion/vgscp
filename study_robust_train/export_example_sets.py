"""Export real prediction sets for individual images, for the qualitative teaser panel.

Why this exists. The paper is about prediction *sets*, but every figure in it reports aggregates.
The one thing a conformal paper can show that nothing else can is a single image with the set each
policy actually returns for it -- marginal split excluding the true label, Mondrian including it,
same model and same scores. That needs per-example membership, which the aggregate CSVs do not
carry.

What this is NOT. It is not a re-run. Nothing is re-trained end-to-end and no features are
re-extracted: the cached features already on Drive are reused, the last-layer head is refitted in
seconds, and the conformal evaluation is the same call the grid makes. The whole thing is an export
of information the pipeline already computes and then discards.

Why it reproduces the paper exactly. Every stochastic step is seeded off ``split_seed`` inside
``conformal_eval.evaluate`` -- the cal/test split, both rho resamplings, and the two uniform streams
that randomise APS -- and the ERM head fits with L-BFGS, whose solver ignores ``random_state``. So
re-deriving a cell gives back the same numbers. Rather than trust that, ``export_example_sets``
looks up the stored CSV row for the same cell and **asserts** the recomputed aggregates match it. If
the figure would disagree with Table 1, this raises instead of writing a file.

Usage on Colab, after the notebook has built ``data[(bb, ds)]`` as usual::

    from study_robust_train.datasets import _load_bundle
    from study_robust_train.export_example_sets import export_example_sets

    import numpy as np
    b = _load_bundle("waterbirds", cfg_for("waterbirds"), 0)          # paths only, cheap
    cat = lambda f: list(f["d_cal"]) + list(f["d_test"])
    export_example_sets(data[("clip_vitb32", "waterbirds")], cat(b.meta["paths"]),
                        "results/calibration_ablation_4bb.csv",
                        backbone="clip_vitb32", dataset="waterbirds",
                        y_eval=np.concatenate([b.y["d_cal"], b.y["d_test"]]),
                        g_eval=np.concatenate([b.group_id["d_cal"], b.group_id["d_test"]]))

The eval pool is ``concatenate(d_cal, d_test)`` in that order (see ``datasets.build_griddata``), and
features are extracted in path order, so index ``i`` of the pool is ``paths_eval[i]``. Passing
``y_eval``/``g_eval`` as above makes that an assertion rather than an assumption: if
``build_griddata`` ever changes the concatenation order, this raises instead of quietly pairing the
wrong image with the right prediction set.
"""
from __future__ import annotations

import csv
import json
import os
import shutil

import numpy as np

from .conformal_eval import evaluate
from .heads import head_probs
from .methods import fit_method

# Waterbirds/CelebA are both binary with group = 2*y + a.
CLASS_NAMES = {
    "waterbirds": ("landbird", "waterbird"),
    "celeba": ("non-blond", "blond"),
}
ATTR_NAMES = {
    "waterbirds": ("land background", "water background"),
    "celeba": ("female", "male"),
}
# Aggregates that must agree with the stored row. Set sizes and coverages are the figure's subject;
# base_top1 is included because it catches a head that refitted differently.
VERIFY = ("marginal_cov", "worst_group_cov", "mean_group_cov", "mean_set_size",
          "worst_group_set_size", "base_top1", "n_eval")


def _stored_row(csv_path, *, backbone, dataset, method, score, calibration, rho_test,
                train_seed, split_seed) -> dict | None:
    want = {"backbone": backbone, "dataset": dataset, "method": method, "score": score,
            "calibration": calibration, "train_seed": str(train_seed),
            "split_seed": str(split_seed)}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if all(row[k] == v for k, v in want.items()) \
                    and abs(float(row["rho_test"]) - rho_test) < 1e-9:
                return row
    return None


def _verify(rec: dict, row: dict, label: str, tol: float = 1e-6) -> None:
    for k in VERIFY:
        got, exp = float(rec[k]), float(row[k])
        if not np.isclose(got, exp, rtol=0, atol=tol):
            raise AssertionError(
                f"[{label}] {k} tidak cocok dengan CSV: dihitung {got!r} vs tersimpan {exp!r}. "
                "Jangan pakai hasil ini untuk gambar -- panel akan bertentangan dengan tabel.")
    print(f"    {label}: {len(VERIFY)} agregat cocok dengan baris CSV tersimpan")


def _describe(membership_row, class_names) -> tuple:
    """Indices in the set, plus a readable rendering. Empty sets are legitimate here (a highly
    accurate model can produce them) and must render as such rather than as ``{}``."""
    idx = [i for i, m in enumerate(membership_row) if m]
    text = "{" + ", ".join(class_names[i] for i in idx) + "}" if idx else "empty set"
    return idx, text


def export_example_sets(gd, paths_eval, csv_path, *, backbone, dataset, method="erm",
                        score="APS", split_seed=0, rho_test=0.95, alpha=0.1, train_seed=0,
                        n_per_case=3, out_dir="results/example_sets",
                        y_eval=None, g_eval=None) -> dict:
    """Refit one head, re-derive both policies' sets, verify against the CSV, export examples.

    ``y_eval``/``g_eval`` are the labels and group ids taken from the *bundle* in the same
    concatenation order as ``paths_eval``. Passing them turns the one unchecked assumption here --
    that pool index ``i`` is ``paths_eval[i]`` -- into an assertion, by confirming the independently
    derived labels agree with the features' own. Without them a silent re-ordering inside
    ``build_griddata`` would put the wrong image next to the right prediction set, which is the one
    error in this file that would be invisible in the output.
    """
    Xev, yev, gev = gd.eval_domain
    if len(paths_eval) != Xev.shape[0]:
        raise ValueError(f"paths_eval ({len(paths_eval)}) != eval pool ({Xev.shape[0]}); "
                         "urutan harus concatenate(d_cal, d_test)")
    if y_eval is None or g_eval is None:
        print("    PERINGATAN: y_eval/g_eval tidak diberikan, jadi korespondensi "
              "indeks->gambar tidak diperiksa. Sangat disarankan mengirimkannya.")
    else:
        if not np.array_equal(np.asarray(y_eval).astype(int), np.asarray(yev).astype(int)):
            raise AssertionError("y dari bundle tidak sama dengan y milik fitur: urutan "
                                 "paths_eval tidak sejajar dengan eval pool")
        if not np.array_equal(np.asarray(g_eval).astype(int), np.asarray(gev).astype(int)):
            raise AssertionError("group dari bundle tidak sama dengan group milik fitur: urutan "
                                 "paths_eval tidak sejajar dengan eval pool")
        print("    korespondensi indeks->gambar terverifikasi (y dan group cocok)")
    classes = CLASS_NAMES[dataset]
    attrs = ATTR_NAMES[dataset]

    print(f"  memasang ulang kepala {method} (seed {train_seed}) pada fitur ter-cache ...")
    head = fit_method(method, gd.train, gd.reweight, seed=train_seed)
    probs = head_probs(head, Xev, gd.n_classes)
    del head

    recs = {}
    for cal in ("marginal_split", "mondrian"):
        rec = evaluate(probs, yev, gev, score=score, alpha=alpha, rho_test=rho_test,
                       split_seed=split_seed, calibration=cal, return_examples=True)
        row = _stored_row(csv_path, backbone=backbone, dataset=dataset, method=method,
                          score=score, calibration=cal, rho_test=rho_test,
                          train_seed=train_seed, split_seed=split_seed)
        if row is None:
            raise LookupError(f"tidak ada baris CSV untuk {cal} pada sel ini; "
                              "periksa backbone/dataset/method/score/seed")
        _verify(rec, row, cal)
        recs[cal] = rec

    m, d = recs["marginal_split"]["examples"], recs["mondrian"]["examples"]
    # Same split_seed means both policies see the identical test rows, which is what makes a
    # per-image comparison meaningful at all.
    if not np.array_equal(m["test_idx"], d["test_idx"]):
        raise AssertionError("baris uji berbeda antar kebijakan; perbandingan per-gambar batal")

    idx, y_t, g_t = m["test_idx"], m["y_test"], m["group_test"]
    cov_m = m["membership"][np.arange(y_t.size), y_t]
    cov_d = d["membership"][np.arange(y_t.size), y_t]
    size_m = m["membership"].sum(axis=1)
    size_d = d["membership"].sum(axis=1)
    worst = int(m["worst_group"])

    # Three cases worth showing, chosen by what they demonstrate rather than at random.
    cases = {
        # the paper's point: Mondrian rescues a worst-group point that marginal drops
        "rescued": np.where((g_t == worst) & (~cov_m) & cov_d)[0],
        # the cost: a worst-group point where Mondrian covers by returning a larger set
        "widened": np.where((g_t == worst) & cov_m & cov_d & (size_d > size_m))[0],
        # the control: a majority point both policies handle identically
        "unchanged": np.where((g_t != worst) & cov_m & cov_d & (size_d == size_m)
                              & (size_m == 1))[0],
    }
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"cell": {"backbone": backbone, "dataset": dataset, "method": method,
                         "score": score, "alpha": alpha, "rho_test": rho_test,
                         "split_seed": split_seed, "train_seed": train_seed},
                "class_names": list(classes), "worst_group": int(worst),
                "thresholds": {"marginal": m["thresholds"], "mondrian": d["thresholds"]},
                "aggregates": {c: {k: float(recs[c][k]) for k in VERIFY} for c in recs},
                "examples": []}

    for case, pool in cases.items():
        print(f"    {case:10s}: {pool.size} kandidat")
        for j in pool[:n_per_case]:
            pool_i = int(idx[j])
            src = paths_eval[pool_i]
            fn = f"{case}_{pool_i}{os.path.splitext(src)[1] or '.jpg'}"
            shutil.copyfile(src, os.path.join(out_dir, fn))
            set_m, txt_m = _describe(m["membership"][j], classes)
            set_d, txt_d = _describe(d["membership"][j], classes)
            manifest["examples"].append({
                "case": case, "file": fn, "pool_index": pool_i,
                "true_class": classes[int(y_t[j])], "true_label": int(y_t[j]),
                "group": int(g_t[j]), "attribute": attrs[int(g_t[j]) % 2],
                "is_worst_group": bool(g_t[j] == worst),
                "marginal": {"set": [classes[i] for i in set_m], "text": txt_m,
                             "size": int(size_m[j]), "covers_truth": bool(cov_m[j])},
                "mondrian": {"set": [classes[i] for i in set_d], "text": txt_d,
                             "size": int(size_d[j]), "covers_truth": bool(cov_d[j])},
                "scores": {classes[i]: float(v)
                           for i, v in enumerate(m["scores_all"][j])},
            })

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"  -> {len(manifest['examples'])} contoh + manifest.json di {out_dir}")
    return manifest
