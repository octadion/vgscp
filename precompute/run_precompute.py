"""One-time precompute over all splits (Section 13).

Runs each frozen model (f, ensemble members, MC-dropout passes, concept extractor + NCV) over
ALL splits ONCE, in large batches under torch.inference_mode() + autocast, caching logits /
probs / features / concepts / p_A terms. Throughput + peak memory are logged per stage.

torch is imported lazily; this module is only exercised on the GPU box.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
from tqdm import tqdm

from perf.setup import PerfContext, autocast_ctx, resolve_num_workers
from perf.throughput import ThroughputLog, stage
from .cache import CacheStore


def _make_loader(dataset, batch_size, perf_cfg, ctx):
    import torch

    dl = perf_cfg.get("dataloader", {})
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,                      # order MUST be stable for caching alignment
        num_workers=resolve_num_workers(dl.get("num_workers", "auto")),
        pin_memory=dl.get("pin_memory", True) and ctx.device == "cuda",
        persistent_workers=dl.get("persistent_workers", True),
        prefetch_factor=dl.get("prefetch_factor", 4),
    )


def precompute_classifier(classifier, dataset, ctx: PerfContext, perf_cfg, tlog, stage_name):
    """Run f over a split once; return (logits, probs, features) as numpy."""
    import torch

    loader = _make_loader(dataset, ctx.batch_size, perf_cfg, ctx)
    logits_all, feats_all = [], []
    with stage(tlog, stage_name, len(dataset)):
        with autocast_ctx(ctx):
            for x, _ in tqdm(loader, desc=stage_name, leave=False):
                x = x.to(ctx.device, non_blocking=True)
                if ctx.channels_last:
                    x = x.to(memory_format=torch.channels_last)
                logits, feats = classifier.logits_and_features(x)
                logits_all.append(logits.float().cpu())
                feats_all.append(feats.float().cpu())
    logits = torch.cat(logits_all).numpy()
    feats = torch.cat(feats_all).numpy()
    probs = _softmax_np(logits)
    return logits, probs, feats


def precompute_member_probs(members, dataset, ctx, perf_cfg, tlog):
    """Run M ensemble members over a split; return (M, N, C)."""
    import torch

    out = []
    for m, clf in enumerate(members):
        loader = _make_loader(dataset, ctx.batch_size, perf_cfg, ctx)
        probs_m = []
        with stage(tlog, f"ensemble_member_{m}", len(dataset)):
            with autocast_ctx(ctx):
                for x, _ in tqdm(loader, desc=f"ens {m}", leave=False):
                    x = x.to(ctx.device, non_blocking=True)
                    if ctx.channels_last:
                        x = x.to(memory_format=torch.channels_last)
                    logits, _ = clf.logits_and_features(x)
                    probs_m.append(torch.softmax(logits.float(), dim=1).cpu())
        out.append(torch.cat(probs_m).numpy())
    return np.stack(out, axis=0)


def precompute_mc_dropout(classifier, dataset, k_passes, ctx, perf_cfg, tlog):
    """Run K batched MC-dropout passes over a split; return (K, N, C)."""
    import torch

    from models.mc_dropout import mc_dropout_probs_batched

    loader = _make_loader(dataset, max(1, ctx.batch_size // k_passes), perf_cfg, ctx)
    chunks = []
    with stage(tlog, "mc_dropout", len(dataset)):
        with autocast_ctx(ctx):
            for x, _ in tqdm(loader, desc="mc-dropout", leave=False):
                x = x.to(ctx.device, non_blocking=True)
                p = mc_dropout_probs_batched(classifier, x, k_passes, ctx)  # (K, b, C)
                chunks.append(p.cpu())
    return torch.cat(chunks, dim=1).numpy()  # concat along N


def precompute_concepts(concept_extractor, dataset, ctx, perf_cfg, tlog, ground_truth=None):
    """Concept encodings g(x). If ground-truth concepts are provided (CLEVR scenes), use them;
    otherwise run a learned concept extractor."""
    if ground_truth is not None:
        return np.asarray(ground_truth)
    import torch

    loader = _make_loader(dataset, ctx.batch_size, perf_cfg, ctx)
    out = []
    with stage(tlog, "concepts", len(dataset)):
        with autocast_ctx(ctx):
            for x, _ in tqdm(loader, desc="concepts", leave=False):
                x = x.to(ctx.device, non_blocking=True)
                out.append(concept_extractor(x).float().cpu())
    return torch.cat(out).numpy()


def precompute_all(
    bundle,
    classifier,
    members,
    verifier,
    ctx: PerfContext,
    perf_cfg: dict,
    cache: CacheStore,
    k_passes: int = 20,
    splits=("train", "d_learn", "d_cal", "d_test"),
    concept_extractor=None,
) -> ThroughputLog:
    """Full one-time precompute over all splits, writing every artifact to the cache."""
    tlog = ThroughputLog()
    for split in splits:
        ds = bundle.datasets[split]
        logits, probs, feats = precompute_classifier(
            classifier, ds, ctx, perf_cfg, tlog, f"f::{split}"
        )
        y_pred = probs.argmax(1)
        cache.save_array(split, "logits", logits)
        cache.save_array(split, "probs", probs)
        cache.save_array(split, "features", feats)
        cache.save_array(split, "y_true", bundle.y[split])
        cache.save_array(split, "y_pred", y_pred)
        cache.save_array(split, "group_id", bundle.group_id[split])
        cache.save_array(split, "spurious_attr", bundle.spurious_attr[split])
        cache.save_array(split, "is_minority", bundle.is_minority[split].astype(np.int64))

        if members:
            cache.save_array(split, "member_probs",
                             precompute_member_probs(members, ds, ctx, perf_cfg, tlog))
        if k_passes:
            cache.save_array(split, "mc_pass_probs",
                             precompute_mc_dropout(classifier, ds, k_passes, ctx, perf_cfg, tlog))

        concepts = precompute_concepts(
            concept_extractor, ds, ctx, perf_cfg, tlog,
            ground_truth=bundle.concepts.get(split),
        )
        cache.save_array(split, "concepts", concepts)

        if verifier is not None:
            with stage(tlog, f"ncv::{split}", len(ds)):
                vo = verifier.predict(concepts, y_pred)
            cache.save_array(split, "pA_given_SM", vo.pA_given_SM)
            cache.save_array(split, "pA_given_SA", vo.pA_given_SA)
            if vo.reject_prob is not None:
                cache.save_array(split, "reject_prob", vo.reject_prob)
            cache.save_json(split, "merlin_concepts", vo.merlin_concepts)
            cache.save_json(split, "morgana_concepts", vo.morgana_concepts)
    return tlog


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
