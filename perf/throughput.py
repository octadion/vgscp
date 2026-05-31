"""Per-stage timing + GPU memory/throughput logging (Section 13).

Confirms GPU utilization is high during the one-time precompute. Records peak memory and
samples/sec per stage; results are written into the run manifest.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StageTiming:
    name: str
    wall_clock_s: float
    n_samples: int
    throughput_sps: float
    peak_mem_mb: Optional[float] = None
    free_mem_mb: Optional[float] = None
    total_mem_mb: Optional[float] = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class ThroughputLog:
    stages: list = field(default_factory=list)

    def add(self, t: StageTiming) -> None:
        self.stages.append(t)

    def to_list(self) -> list:
        return [s.to_dict() for s in self.stages]


@contextmanager
def stage(log: ThroughputLog, name: str, n_samples: int):
    """Time a stage and capture peak GPU memory; appends a StageTiming to ``log``."""
    torch = None
    try:
        import torch as _torch

        torch = _torch
    except ImportError:
        pass

    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    yield
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    peak = free = total = None
    if torch is not None and torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024**2)
        free_b, total_b = torch.cuda.mem_get_info()
        free = free_b / (1024**2)
        total = total_b / (1024**2)

    log.add(
        StageTiming(
            name=name,
            wall_clock_s=dt,
            n_samples=int(n_samples),
            throughput_sps=(n_samples / dt if dt > 0 else 0.0),
            peak_mem_mb=peak,
            free_mem_mb=free,
            total_mem_mb=total,
        )
    )
