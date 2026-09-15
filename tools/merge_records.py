"""Merge record CSVs produced by two Colab runtimes into one file.

The streaming runs rewrite their whole CSV after each finished cell, so two runtimes cannot share
one file. They write halves instead (``*_part2.csv``), and this puts the halves back together.

    python tools/merge_records.py ablation  results/calibration_ablation_4bb.csv  a.csv b.csv
    python tools/merge_records.py predgroup results/predicted_group_mondrian_4bb.csv  a.csv b.csv

Rows are merged as text, so no value is re-formatted on the way through. Three things make it
refuse rather than produce a plausible-looking wrong file:

  * a row present in both inputs with DIFFERENT numbers -- that means the two runtimes disagree,
    which after the label-conditional set fix most likely means one clone predates it;
  * a missing (backbone, dataset) cell, unless --allow-partial says a half-finished merge is
    wanted on purpose;
  * inputs whose headers are not identical.

An exactly duplicated row is dropped and counted, so a cell repeated by both runtimes is harmless.
"""
from __future__ import annotations

import csv
import io
import sys

KEYS = {
    "ablation": ("backbone", "dataset", "method", "train_seed", "calibration", "score",
                 "rho_cal", "rho_test", "split_seed"),
    "predgroup": ("backbone", "dataset", "method", "train_seed", "condition", "score",
                  "rho_cal", "rho_test", "split_seed"),
}
BACKBONES = ("resnet50_erm", "clip_vitb32", "dinov2_vitb14", "vit_b16_in1k")
DATASETS = ("waterbirds", "celeba")


def read(path):
    with io.open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def main(argv) -> int:
    allow_partial = "--allow-partial" in argv
    argv = [a for a in argv if a != "--allow-partial"]
    if len(argv) < 4 or argv[1] not in KEYS:
        print(__doc__)
        return 2
    study, out_path, inputs = argv[1], argv[2], argv[3:]
    key_cols = KEYS[study]

    header, merged, dup_same, conflicts = None, {}, 0, []
    for path in inputs:
        cols, rows = read(path)
        if header is None:
            header = cols
        elif cols != header:
            print(f"[STOP] {path} has a different header than {inputs[0]}")
            return 1
        for row in rows:
            k = tuple(row[c] for c in key_cols)
            prev = merged.get(k)
            if prev is None:
                merged[k] = (path, row)
            elif prev[1] == row:
                dup_same += 1
            else:
                differing = sorted(c for c in header if prev[1].get(c) != row.get(c))
                conflicts.append((k, prev[0], path, differing))
        print(f"  read {len(rows):7,} rows from {path}")

    if conflicts:
        print(f"\n[STOP] {len(conflicts)} row(s) present in two inputs with different values.")
        for k, p1, p2, cols in conflicts[:5]:
            print(f"   {k}\n      {p1} vs {p2}: {cols}")
        print("   Nothing written. Check that both runtimes ran the same commit.")
        return 1

    cells = {}
    for _, row in merged.values():
        cells[(row["backbone"], row["dataset"])] = cells.get((row["backbone"], row["dataset"]), 0) + 1
    missing = [(bb, ds) for ds in DATASETS for bb in BACKBONES if (bb, ds) not in cells]

    print(f"\n  merged {len(merged):,} unique rows ({dup_same:,} exact duplicate(s) dropped)")
    print(f"  cells  {len(cells)}/8")
    for ds in DATASETS:
        for bb in BACKBONES:
            n = cells.get((bb, ds))
            print(f"     {ds:11s} {bb:14s} {n:7,} rows" if n else
                  f"     {ds:11s} {bb:14s}       -- MISSING")
    sizes = {n for n in cells.values()}
    if len(sizes) > 1:
        print(f"  [note] cells do not all hold the same number of rows: {sorted(sizes)}")

    if missing and not allow_partial:
        print(f"\n[STOP] {len(missing)} cell(s) missing: {missing}")
        print("   Nothing written. Finish those cells, or pass --allow-partial on purpose.")
        return 1

    with io.open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for _, row in sorted(merged.items()):
            w.writerow(row[1])
    print(f"\n  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
