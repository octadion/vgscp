"""Pre-flight audit of the representation run against the code it will actually call.

Catches the class of bug that cost three Waterbirds re-runs and one aborted CelebA run: a config
value the callee silently ignores, an epoch budget shared across datasets whose train splits differ
34x in size, or a knob that changes the learned representation without changing the cache key (so a
re-run answers with stale features and the change looks inert).

Two entry points, and the difference matters:

  ``audit(globals())``   audits the **live** notebook namespace -- the values actually in scope,
                         including anything edited in the browser. Call it from the notebook after
                         the definitions cell and before any training cell.

  ``python -m study_robust_train.preflight_representation``
                         standalone: reads the committed notebook, executes only its config cell,
                         and audits that. Convenient locally, but it checks the file on disk, not
                         whatever the running notebook currently holds.

Static and instant either way: no training cell is ever executed.
"""
from __future__ import annotations

import ast
import inspect
import itertools
import json
import os
import sys

from .finetune import OBJECTIVES, cache_key, finetune_features

# Measured on an L4 at batch_size=128 with AMP; used only for the cost projection.
IMG_PER_SEC = 484.0
EXTRACT_N = {"waterbirds": 12040, "celeba": 202599}
TRAIN_N = {"waterbirds": 4795, "celeba": 162770}
DRIVE_BUDGET_GB = 13.0
SESSION_HOURS = 4.0

REQUIRED = ("FT_OBJECTIVES", "FT_SEEDS", "FT_EPOCHS", "FT_HP", "BATCH_SIZE", "CACHE_DTYPE")


def per_gd_celeba() -> float:
    """GB held by one CelebA GridData (train + reweight + eval_domain features, float32)."""
    return (162770 + 9957 + 29872) * 2048 * 4 / 2**30


def audit(ns: dict, *, verbose: bool = True) -> bool:
    """Audit a live namespace (or any dict of config values). Returns True if clean."""
    fails: list[str] = []

    def chk(name, cond, detail=""):
        if verbose:
            print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
        if not cond:
            fails.append(name)

    def sec(t):
        if verbose:
            print(f"\n[{t}]")

    sec("config present")
    for k in REQUIRED:
        chk(f"{k} defined", k in ns, repr(ns.get(k))[:64])
    if fails:
        print("\nmissing config -- run the parameters cell first")
        return False

    objs, eps, hp = ns["FT_OBJECTIVES"], ns["FT_EPOCHS"], ns["FT_HP"]
    wb_seeds = tuple(ns["FT_SEEDS"])
    cel_seeds = tuple(ns.get("CELEBA_SEEDS", wb_seeds))
    max_train = ns.get("CELEBA_MAX_TRAIN")

    sec("epoch budgets (the defect that cost ~12 h)")
    chk("objectives are all real", set(objs) <= set(OBJECTIVES), f"{objs} vs {OBJECTIVES}")
    for ds in ("waterbirds", "celeba"):
        missing = [o for o in objs if (ds, o) not in eps]
        chk(f"{ds}: every objective has an epoch budget", not missing, f"missing={missing}")
    if all((("celeba", o) in eps) for o in objs):
        worst = max(eps[("celeba", o)] for o in objs)
        chk("CelebA epoch budgets are dataset-appropriate (<=10)", worst <= 10,
            f"max {worst} epochs; one full-train CelebA epoch is ~5.7 min")
    chk("CELEBA_MAX_TRAIN is set explicitly", max_train is not None,
        "None means all 162,770 images -- ~3.25x the cost")

    sec("no config key is silently ignored")
    sig = set(inspect.signature(finetune_features).parameters)
    hp_keys = set(itertools.chain.from_iterable(d.keys() for d in hp.values()))
    chk("every FT_HP key is a real parameter", hp_keys <= sig, f"unknown={sorted(hp_keys - sig)}")
    chk("no stale epochs_override in FT_HP", not any("epochs_override" in d for d in hp.values()))
    # Source-level checks are best-effort: inspect.getsource cannot recover the text of a function
    # defined by bare exec. A gate must never abort a healthy run, so an unavailable source is
    # reported as skipped rather than raised or silently passed.
    if "cfg_for" in ns:
        try:
            text = inspect.getsource(ns["cfg_for"])
        except (OSError, TypeError):
            text = None
        if text is None:
            if verbose:
                print("  SKIP  cfg_for key check -- source unavailable for this namespace")
        else:
            keys = set()
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, ast.Dict):
                    keys |= {k.value for k in node.keys
                             if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            keys -= {"root", "image_size", "n_classes", "download", "dataset", "finetune"}
            chk("every cfg_for key is a real parameter", keys <= sig,
                f"unknown={sorted(keys - sig)}")
    for fn, want in (("keys_for", {"dataset", "seeds"}), ("builder", {"dataset", "max_train"})):
        if fn not in ns:
            continue
        try:
            p = set(inspect.signature(ns[fn]).parameters)
        except (TypeError, ValueError):
            if verbose:
                print(f"  SKIP  {fn} signature check -- not introspectable")
            continue
        chk(f"{fn} takes {sorted(want)}", want <= p, f"has {sorted(p)}")

    sec("host RAM (the crash that killed a CelebA run)")
    # A DataLoader reserves batch x workers x prefetch DECODED images in host RAM. At 224x224x3
    # float32 that is 0.574 MB each, so the product -- not the batch size alone -- is the budget.
    from .finetune import EXTRACT_INFLIGHT_BYTES
    img = 3 * 224 * 224 * 4
    ebs = ns.get("EXTRACT_BS") or ns["BATCH_SIZE"]
    budget = ns.get("EXTRACT_INFLIGHT") or EXTRACT_INFLIGHT_BYTES
    nw = ns.get("NUM_WORKERS") or 12          # None resolves to <=12 at runtime
    train_inflight = ns["BATCH_SIZE"] * nw * 4 * img / 2**30      # training uses prefetch_factor=4
    ext_workers = max(2, min(nw, int(budget / (ebs * 2 * img))))
    ext_inflight = ebs * ext_workers * 2 * img / 2**30
    if verbose:
        print(f"        training loader : bs={ns['BATCH_SIZE']} x {nw} workers x prefetch 4"
              f"  -> {train_inflight:.1f} GB in flight")
        print(f"        extraction loader: bs={ebs} x {ext_workers} workers (derived) x prefetch 2"
              f"  -> {ext_inflight:.1f} GB in flight")
    peak = ext_inflight + per_gd_celeba() + 1.24 + 2.0
    if verbose:
        print(f"        projected peak with streaming: {peak:.1f} GB "
              f"(standard Colab ~12.7, high-RAM ~51)")
    chk("projected peak RAM fits a standard runtime", peak <= 11.0, f"{peak:.1f} GB")
    chk("training in-flight RAM within budget", train_inflight <= 4.0, f"{train_inflight:.1f} GB")
    # Streaming keeps ONE GridData live; the batched form keeps them all, which is ~14 GB on CelebA.
    per_gd = per_gd_celeba()
    n_cel = len(objs) * len(cel_seeds)
    if verbose:
        print(f"        one CelebA GridData: {per_gd:.2f} GB   "
              f"| all {n_cel} at once: {per_gd*n_cel:.1f} GB (streaming avoids this)")
    # Only assertable once the definitions cell has run; the standalone path execs the config cell
    # alone, so there is nothing to inspect there.
    if "cfg_for" in ns:
        chk("CelebA analysis is streamed, not accumulated",
            "run_representation_streaming" in ns,
            "cell 8 must use run_representation_streaming, not hold every GridData")
    elif verbose:
        print("  SKIP  streaming check -- run this from the notebook gate to assert it")

    sec("representation-affecting knobs are all keyed")
    base_kw = dict(epochs=10, seed=0, max_train=None, lr=1e-3, batch_size=128,
                   optimizer="sgd", weight_decay=1e-2, groupdro_eta=0.05)
    P = {"train": ["p"]}
    base = cache_key("groupdro", "waterbirds", P, **base_kw)
    for name, over in (("epochs", {"epochs": 99}), ("lr", {"lr": 9e-9}),
                       ("batch_size", {"batch_size": 999}), ("optimizer", {"optimizer": "adam"}),
                       ("weight_decay", {"weight_decay": 0.5}),
                       ("groupdro_eta", {"groupdro_eta": 0.9}), ("seed", {"seed": 7}),
                       ("max_train", {"max_train": 123}),
                       ("init_weights", {"init_weights": "IMAGENET1K_V1"}),
                       ("amp", {"amp": False})):
        chk(f"changing {name} changes the cache key",
            cache_key("groupdro", "waterbirds", P, **{**base_kw, **over}) != base)
    ck = set(inspect.signature(cache_key).parameters)
    chk("extract_batch_size is NOT keyed (no-grad forward, values identical)",
        "extract_batch_size" not in ck)
    src = inspect.getsource(finetune_features)
    # Neither is a hyperparameter, so neither is in cache_key -- which is why both must stay off:
    # arms computed under different settings are not comparable and the cache would not show it.
    chk("TF32 disabled (changes precision outside the cache key)",
        "allow_tf32 = False" in src and "allow_tf32 = True" not in src)
    chk("cudnn.benchmark disabled (changes kernel selection outside the cache key)",
        "cudnn.benchmark = False" in src and "cudnn.benchmark = True" not in src)

    sec("predicted cache keys")
    def key(ds, obj, seed, mt):
        h = hp.get(obj, {})
        return cache_key(obj, ds, P, epochs=eps[(ds, obj)], seed=seed, max_train=mt,
                         lr=h.get("lr", 1e-3), batch_size=ns["BATCH_SIZE"],
                         optimizer=h.get("optimizer", "adam"),
                         weight_decay=h.get("weight_decay", 0.0),
                         groupdro_eta=h.get("groupdro_eta", 0.01),
                         amp=ns.get("AMP", True))
    if verbose:
        for ds, mt in (("waterbirds", None), ("celeba", max_train)):
            for o in objs:
                print(f"        {ds}/{o}/s0 -> ...{key(ds, o, 0, mt).split('_', 3)[3]}")
    # The last good Waterbirds run produced this suffix; if it still matches, that cache is reused.
    chk("Waterbirds erm/s0 key matches the last good run -> cache HIT",
        key("waterbirds", "erm", 0, None).endswith(
            "in1kV2_10ep_lr0.001_bs128_wd0_adam_mtall_s0"))
    if max_train:
        chk("CelebA keys carry the subsample -> the aborted full-train run is not reused",
            f"mt{int(max_train)}" in key("celeba", "erm", 0, max_train))

    sec("cost and Drive projection")
    # Only meaningful once every budget exists; a missing one is already reported above, and
    # projecting through it would raise and hide the remaining checks.
    complete = all((ds, o) in eps for ds in ("waterbirds", "celeba") for o in objs)
    if not complete:
        if verbose:
            print("  SKIP  projection -- fill in the missing epoch budgets first")
        if verbose:
            print("\n" + "=" * 74)
            print(f"PRE-FLIGHT FAILED ({len(fails)}): " + ", ".join(fails))
        return False
    per_img = 2 if ns["CACHE_DTYPE"] == "float16" else 4
    total_h, cel_h, drive = 0.0, 0.0, 0.0
    for ds, seeds in (("waterbirds", wb_seeds), ("celeba", cel_seeds)):
        sub = 0.0
        n_train = TRAIN_N[ds] if ds == "waterbirds" else min(TRAIN_N[ds], max_train or TRAIN_N[ds])
        for o in objs:
            per = (n_train * eps[(ds, o)] / IMG_PER_SEC + EXTRACT_N[ds] / 1200) / 60
            sub += per * len(seeds)
            if verbose:
                print(f"        {ds:10s} {o:9s} {eps[(ds,o)]:2d}ep x{len(seeds)} seeds  "
                      f"{per:5.1f} min/run -> {per*len(seeds):6.1f} min")
        drive += EXTRACT_N[ds] * 2048 * per_img / 1e9 * len(seeds) * len(objs)
        total_h += sub / 60
        if ds == "celeba":
            cel_h = sub / 60
    if verbose:
        print(f"\n        all runs from scratch : {total_h:.1f} h")
        print(f"        CelebA only           : {cel_h:.1f} h  (Waterbirds cached)")
        print(f"        Drive feature caches  : {drive:.1f} GB  (CACHE_DTYPE={ns['CACHE_DTYPE']})")
    chk(f"CelebA fits one {SESSION_HOURS:g} h session", cel_h < SESSION_HOURS, f"{cel_h:.1f} h")
    chk(f"Drive need under {DRIVE_BUDGET_GB:g} GB", drive < DRIVE_BUDGET_GB, f"{drive:.1f} GB")

    if verbose:
        print("\n" + "=" * 74)
        print(f"PRE-FLIGHT FAILED ({len(fails)}): " + ", ".join(fails) if fails
              else "PRE-FLIGHT CLEAN")
    return not fails


def _from_notebook(path: str) -> dict:
    """Execute only the notebook's config cell and return its namespace."""
    nb = json.load(open(path, encoding="utf-8"))
    cells = [("".join(c["source"]), c["cell_type"]) for c in nb["cells"]]
    cfg_cell = next(s for s, t in cells if t == "code" and "FT_OBJECTIVES" in s)
    ns: dict = {}
    exec(cfg_cell, ns)
    return ns


def main() -> int:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nb = os.environ.get("REPR_NB", os.path.join(repo, "notebooks",
                                                "representation_finetune.ipynb"))
    if not os.path.exists(nb):
        print(f"notebook not found: {nb}\nset REPR_NB=/path/to/representation_finetune.ipynb")
        return 1
    print(f"auditing committed notebook: {nb}")
    print("(the notebook gate audits the LIVE namespace instead -- see module docstring)")
    return 0 if audit(_from_notebook(nb)) else 1


if __name__ == "__main__":
    sys.exit(main())
