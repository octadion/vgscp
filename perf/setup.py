"""Global performance / determinism / precision setup (Section 13 of the spec).

Set ONCE at the start of every run and applied everywhere. The precision and batch policy
must NOT change mid-suite — fairness across methods depends on identical compute.

This module imports torch lazily so that the pure-numpy ``theory/`` testbed can run on a
machine without torch installed.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# --------------------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------------------
def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed python / numpy / torch and (optionally) enable deterministic algorithms."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            # cudnn determinism; some ops fall back to slower deterministic kernels.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Required for deterministic algorithms with some conv/cublas ops.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    except ImportError:
        pass


# --------------------------------------------------------------------------------------
# Device / precision resolution
# --------------------------------------------------------------------------------------
@dataclass
class PerfContext:
    """Resolved runtime perf state. Logged into the run manifest for reproducibility."""

    device: str = "cpu"
    gpu_name: str = ""
    precision: str = "fp32"        # resolved: bf16 | fp16 | fp32
    amp_dtype: Optional[Any] = None  # torch dtype or None
    use_amp: bool = False
    channels_last: bool = False
    inference_mode: bool = True
    batch_size: int = 0            # resolved (possibly via VRAM probe)
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    deterministic: bool = True
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["amp_dtype"] = str(self.amp_dtype)
        return d


def _supports_bf16(torch) -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_bf16_supported()
    except Exception:
        # Compute capability >= 8.0 (Ampere+) supports bf16; Turing (7.5, T4) does not.
        major, _ = torch.cuda.get_device_capability()
        return major >= 8


def resolve_precision(requested: str, torch) -> tuple[str, Any]:
    """Resolve 'auto' -> bf16 (Ampere+/L4) else fp16 (Turing/T4); cpu -> fp32."""
    if not torch.cuda.is_available():
        return "fp32", None
    if requested == "auto":
        prec = "bf16" if _supports_bf16(torch) else "fp16"
    else:
        prec = requested
    if prec == "fp32":
        return "fp32", None
    if prec == "bf16":
        return "bf16", torch.bfloat16
    if prec == "fp16":
        return "fp16", torch.float16
    raise ValueError(f"unknown precision: {requested}")


def setup_perf(perf_cfg: dict, seed: int = 0) -> PerfContext:
    """Apply the shared perf config globally and return the resolved PerfContext.

    Safe to call without torch (returns a cpu context with fp32) so the theory module and
    unit tests run anywhere.
    """
    deterministic = bool(perf_cfg.get("deterministic", True))
    if perf_cfg.get("seed_everything", True):
        seed_everything(seed, deterministic=deterministic)

    try:
        import torch
    except ImportError:
        return PerfContext(device="cpu", precision="fp32", deterministic=deterministic)

    ctx = PerfContext(deterministic=deterministic)
    ctx.allow_tf32 = bool(perf_cfg.get("allow_tf32", True))
    ctx.cudnn_benchmark = bool(perf_cfg.get("cudnn_benchmark", True)) and not deterministic
    ctx.channels_last = bool(perf_cfg.get("channels_last", True))
    ctx.inference_mode = bool(perf_cfg.get("inference_mode", True))
    ctx.use_amp = bool(perf_cfg.get("use_amp", True))

    if torch.cuda.is_available():
        ctx.device = "cuda"
        ctx.gpu_name = torch.cuda.get_device_name(0)
        torch.backends.cuda.matmul.allow_tf32 = ctx.allow_tf32
        torch.backends.cudnn.allow_tf32 = ctx.allow_tf32
        if not deterministic:
            torch.backends.cudnn.benchmark = ctx.cudnn_benchmark
        prec, amp_dtype = resolve_precision(perf_cfg.get("precision", "auto"), torch)
        ctx.precision = prec
        ctx.amp_dtype = amp_dtype
        # On CPU AMP is pointless; on fp32 path amp is a no-op.
        ctx.use_amp = ctx.use_amp and amp_dtype is not None
    else:
        ctx.device = "cpu"
        ctx.precision = "fp32"
        ctx.use_amp = False
    return ctx


def autocast_ctx(ctx: PerfContext):
    """Context manager pairing inference_mode + autocast per the resolved PerfContext."""
    import contextlib

    import torch

    managers = []
    if ctx.inference_mode:
        managers.append(torch.inference_mode())
    if ctx.use_amp and ctx.device == "cuda" and ctx.amp_dtype is not None:
        managers.append(torch.autocast(device_type="cuda", dtype=ctx.amp_dtype))

    @contextlib.contextmanager
    def _combined():
        with contextlib.ExitStack() as stack:
            for m in managers:
                stack.enter_context(m)
            yield

    return _combined()


def resolve_num_workers(cfg_value) -> int:
    if cfg_value == "auto" or cfg_value is None:
        return max(1, (os.cpu_count() or 2))
    return int(cfg_value)
