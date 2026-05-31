"""VRAM probe (Section 13).

Auto-increase batch size until ~target_frac of *free* VRAM is used, then FIX it. The same
batch policy is reused for every experiment in the suite (fairness — never change it mid-suite).
If a stage OOMs, reduce batch via this probe — never by changing precision/seed.
"""
from __future__ import annotations

from typing import Callable, Optional

from .setup import PerfContext, autocast_ctx


def _classify_gpu(name: str) -> str:
    n = name.lower()
    if "t4" in n:
        return "t4"
    if "l4" in n:
        return "l4"
    return "default"


def fallback_batch(perf_cfg: dict, gpu_name: str) -> int:
    p = perf_cfg.get("vram_probe", {})
    kind = _classify_gpu(gpu_name)
    return int(
        {
            "t4": p.get("fallback_t4", 256),
            "l4": p.get("fallback_l4", 512),
            "default": p.get("fallback_default", 128),
        }[kind]
    )


def probe_batch_size(
    ctx: PerfContext,
    perf_cfg: dict,
    make_batch: Callable[[int], "object"],
    forward: Callable[["object"], "object"],
) -> int:
    """Binary-doubling probe.

    ``make_batch(bs)`` builds a representative dummy batch on the device; ``forward(batch)``
    runs the heaviest forward pass for the stage. We double until OOM or until we exceed
    target_frac of free memory, then return the largest batch that fit.
    """
    import torch

    pcfg = perf_cfg.get("vram_probe", {})
    if not pcfg.get("enabled", True) or ctx.device != "cuda":
        bs = fallback_batch(perf_cfg, ctx.gpu_name)
        ctx.batch_size = bs
        return bs

    target_frac = float(pcfg.get("target_vram_frac", 0.90))
    bs = int(pcfg.get("min_batch", 8))
    max_bs = int(pcfg.get("max_batch", 1024))
    best = bs

    while bs <= max_bs:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            batch = make_batch(bs)
            with autocast_ctx(ctx):
                _ = forward(batch)
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            used = total - free
            frac = used / total
            best = bs
            if frac >= target_frac:
                break
            bs *= 2
        except RuntimeError as e:  # OOM
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                break
            raise
    # back off one notch from the largest that fit, to leave headroom for fragmentation
    chosen = max(int(pcfg.get("min_batch", 8)), int(best * 0.9))
    ctx.batch_size = chosen
    torch.cuda.empty_cache()
    return chosen
