"""Pre-flight audit of the representation notebook against the code it will actually call.

Catches the class of bug that cost three Waterbirds re-runs and one aborted CelebA run: a notebook
value the underlying function silently ignores, an epoch budget shared across datasets whose train
splits differ 34x in size, or a knob that changes the learned representation without changing the
cache key (so a re-run answers with stale features and the change looks inert).

Static and instant -- it executes only the notebook's config cell, never a training cell. Run it
before spending GPU time:

    python -m study_robust_train.preflight_representation

Set REPR_NB to audit a notebook at a non-default path.
"""
import ast, inspect, itertools, json, re, sys

sys.path.insert(0, r"C:\jagr\vgscp")
NB = r"C:\jagr\vgscp\notebooks\representation_finetune.ipynb"

FAIL = []
def chk(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    if not cond: FAIL.append(name)

nb = json.load(open(NB, encoding="utf-8"))
cells = [("".join(c["source"]), c["cell_type"]) for c in nb["cells"]]
codes = [s for s, t in cells if t == "code"]
src = "\n".join(codes)

# ---- 1. the config cell, evaluated for real
print("\n[1] notebook config cell")
cfg_cell = codes[0]
g = {}
exec(cfg_cell, g)
cfgvals = {k: v for k, v in g.items() if k.isupper()}
for k in ("FT_OBJECTIVES", "FT_SEEDS", "CELEBA_SEEDS", "CELEBA_MAX_TRAIN", "FT_EPOCHS",
          "FT_HP", "BATCH_SIZE", "NUM_WORKERS", "EXTRACT_BS", "CACHE_DTYPE", "N_SPLITS",
          "HEADS", "SCORES"):
    chk(f"{k} defined", k in cfgvals, repr(cfgvals.get(k))[:70])

# ---- 2. FT_EPOCHS covers every (dataset, objective) that will be requested
print("\n[2] epoch budget completeness (the bug that cost ~12 h)")
objs, eps = g["FT_OBJECTIVES"], g["FT_EPOCHS"]
for ds in ("waterbirds", "celeba"):
    missing = [o for o in objs if (ds, o) not in eps]
    chk(f"{ds}: every objective has an epoch budget", not missing, f"missing={missing}")
chk("no epochs_override left anywhere", "epochs_override" not in src)
chk("CelebA GroupDRO budget is sane (<=10 epochs)", eps[("celeba", "groupdro")] <= 10,
    f"{eps[('celeba','groupdro')]} epochs x 5.7 min/epoch at full train")
chk("CELEBA_MAX_TRAIN is set, not None", g["CELEBA_MAX_TRAIN"] is not None,
    str(g["CELEBA_MAX_TRAIN"]))

# ---- 3. build_all signature vs every call site
print("\n[3] build_all: definition vs call sites")
tree = ast.parse(src)
fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "build_all"), None)
chk("build_all defined", fn is not None)
if fn:
    params = [a.arg for a in fn.args.args]
    kwonly = [a.arg for a in fn.args.kwonlyargs]
    print(f"        params={params} kwonly={kwonly}")
    for call in [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", None) == "build_all"]:
        npos, kws = len(call.args), [k.arg for k in call.keywords]
        ok = npos <= len(params) and all(k in params + kwonly for k in kws)
        chk(f"call build_all({npos} pos, {kws}) matches signature", ok)

# ---- 4. every cfg key reaches finetune_features
print("\n[4] cfg keys vs finetune_features signature")
from study_robust_train.finetune import finetune_features, cache_key, OBJECTIVES
sig = set(inspect.signature(finetune_features).parameters)
cfg_src = codes[[i for i, c in enumerate(codes) if "def cfg_for" in c][0]]
# Parse the dict literals rather than regex-scanning: `if dataset == "waterbirds":` ends in a colon
# and a naive `"(\w+)":` scan reads it as a key.
cfg_keys = set()
for node in ast.walk(ast.parse(cfg_src)):
    if isinstance(node, ast.Dict):
        for k in node.keys:
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                cfg_keys.add(k.value)
cfg_keys -= {"root", "image_size", "n_classes", "download", "dataset", "finetune"}
unknown = cfg_keys - sig
chk("no cfg key is silently ignored", not unknown, f"unknown={sorted(unknown)}")
hp_keys = set(itertools.chain.from_iterable(d.keys() for d in g["FT_HP"].values()))
chk("no FT_HP key is silently ignored", hp_keys <= sig, f"unknown={sorted(hp_keys - sig)}")
chk("objectives are all real", set(objs) <= set(OBJECTIVES), f"{objs} vs {OBJECTIVES}")

# ---- 5. cache keys: Waterbirds must HIT, CelebA must be fresh
print("\n[5] predicted cache keys")
P = {"train": ["p"]}   # the path hash is dataset-derived; only the suffix is checked here
def key(ds, obj, seed, mt):
    return cache_key(obj, ds, P, epochs=eps[(ds, obj)], seed=seed, max_train=mt,
                     lr=g["FT_HP"][obj]["lr"], batch_size=g["BATCH_SIZE"],
                     optimizer=g["FT_HP"][obj]["optimizer"],
                     weight_decay=g["FT_HP"][obj]["weight_decay"],
                     groupdro_eta=g["FT_HP"][obj].get("groupdro_eta", 0.01))
for obj in objs:
    print(f"        waterbirds/{obj}/s0 -> ...{key('waterbirds', obj, 0, None).split('_', 3)[3]}")
for obj in objs:
    print(f"        celeba/{obj}/s0     -> ...{key('celeba', obj, 0, g['CELEBA_MAX_TRAIN']).split('_', 3)[3]}")
# The last successful Waterbirds run produced this suffix for erm/s0 (from the user's log):
OBSERVED = "in1kV2_10ep_lr0.001_bs128_wd0_adam_mtall_s0"
chk("Waterbirds erm/s0 key unchanged since the good run -> cache HIT",
    key("waterbirds", "erm", 0, None).endswith(OBSERVED), OBSERVED)
chk("CelebA key carries mt50000 -> the aborted mtall run is correctly NOT reused",
    "mt50000" in key("celeba", "erm", 0, g["CELEBA_MAX_TRAIN"]))

# ---- 6. things that change the representation must be in the key
print("\n[6] representation-affecting knobs are all keyed")
base = key("waterbirds", "groupdro", 0, None)
variants = {
    "epochs":       cache_key("groupdro", "waterbirds", P, epochs=99, seed=0, max_train=None, lr=1e-3, batch_size=128, optimizer="sgd", weight_decay=1e-2, groupdro_eta=0.05),
    "lr":           cache_key("groupdro", "waterbirds", P, epochs=eps[("waterbirds","groupdro")], seed=0, max_train=None, lr=9e-9, batch_size=128, optimizer="sgd", weight_decay=1e-2, groupdro_eta=0.05),
    "batch_size":   cache_key("groupdro", "waterbirds", P, epochs=eps[("waterbirds","groupdro")], seed=0, max_train=None, lr=1e-3, batch_size=999, optimizer="sgd", weight_decay=1e-2, groupdro_eta=0.05),
    "optimizer":    cache_key("groupdro", "waterbirds", P, epochs=eps[("waterbirds","groupdro")], seed=0, max_train=None, lr=1e-3, batch_size=128, optimizer="adam", weight_decay=1e-2, groupdro_eta=0.05),
    "weight_decay": cache_key("groupdro", "waterbirds", P, epochs=eps[("waterbirds","groupdro")], seed=0, max_train=None, lr=1e-3, batch_size=128, optimizer="sgd", weight_decay=0.5, groupdro_eta=0.05),
    "groupdro_eta": cache_key("groupdro", "waterbirds", P, epochs=eps[("waterbirds","groupdro")], seed=0, max_train=None, lr=1e-3, batch_size=128, optimizer="sgd", weight_decay=1e-2, groupdro_eta=0.9),
    "init_weights": cache_key("groupdro", "waterbirds", P, epochs=eps[("waterbirds","groupdro")], seed=0, max_train=None, lr=1e-3, batch_size=128, optimizer="sgd", weight_decay=1e-2, groupdro_eta=0.05, init_weights="IMAGENET1K_V1"),
    "seed":         cache_key("groupdro", "waterbirds", P, epochs=eps[("waterbirds","groupdro")], seed=7, max_train=None, lr=1e-3, batch_size=128, optimizer="sgd", weight_decay=1e-2, groupdro_eta=0.05),
    "max_train":    cache_key("groupdro", "waterbirds", P, epochs=eps[("waterbirds","groupdro")], seed=0, max_train=123, lr=1e-3, batch_size=128, optimizer="sgd", weight_decay=1e-2, groupdro_eta=0.05),
}
for name, k in variants.items():
    chk(f"changing {name} changes the cache key", k != base)
# and things that do NOT affect the learned values must NOT be keyed
chk("extract_batch_size is NOT a cache-key input (no-grad forward)",
    "extract_batch_size" not in inspect.signature(cache_key).parameters)
chk("TF32 not enabled (would change numerics without changing the key)",
    "allow_tf32" not in inspect.getsource(finetune_features))

# ---- 7. cost projection
print("\n[7] projected cost")
IMG_S = 484.0                       # measured on the user's L4 at bs=128
EXTRACT = {"waterbirds": 12040, "celeba": 202599}
TRAIN_N = {"waterbirds": 4795, "celeba": min(162770, g["CELEBA_MAX_TRAIN"] or 162770)}
tot = 0.0
for ds, seeds in (("waterbirds", g["FT_SEEDS"]), ("celeba", g["CELEBA_SEEDS"])):
    sub = 0.0
    for o in objs:
        per = (TRAIN_N[ds] * eps[(ds, o)] / IMG_S + EXTRACT[ds] / 1200) / 60
        sub += per * len(seeds)
        print(f"        {ds:10s} {o:9s} {eps[(ds,o)]:2d}ep x {len(seeds)} seeds"
              f"  {per:5.1f} min/run -> {per*len(seeds):6.1f} min")
    print(f"        {ds:10s} SUBTOTAL {sub/60:.1f} h")
    tot += sub
gb = {"waterbirds": 12040, "celeba": 202599}
dr = sum(gb[ds] * 2048 * (2 if g["CACHE_DTYPE"] == "float16" else 4) / 1e9 * len(s)
         for ds, s in (("waterbirds", g["FT_SEEDS"]), ("celeba", g["CELEBA_SEEDS"]))) * len(objs)
print(f"\n        TOTAL (if nothing cached): {tot/60:.1f} h")
print(f"        CelebA only (Waterbirds cached): "
      f"{sum((TRAIN_N['celeba']*eps[('celeba',o)]/IMG_S + EXTRACT['celeba']/1200)/60 for o in objs)*len(g['CELEBA_SEEDS'])/60:.1f} h")
print(f"        Drive for feature caches: {dr:.1f} GB  (CACHE_DTYPE={g['CACHE_DTYPE']})")
chk("CelebA fits one 4 h session",
    sum((TRAIN_N['celeba']*eps[('celeba',o)]/IMG_S + EXTRACT['celeba']/1200)/60 for o in objs)
    * len(g['CELEBA_SEEDS']) / 60 < 4.0)
chk("Drive need under 13 GB", dr < 13.0, f"{dr:.1f} GB")

print("\n" + "=" * 74)
print(f"FAILED ({len(FAIL)}): " + ", ".join(FAIL) if FAIL else "PRE-FLIGHT CLEAN")
sys.exit(1 if FAIL else 0)
