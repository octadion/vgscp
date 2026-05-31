"""Waterbirds loader — PHASE 2 scope (generalization benchmark). Stub for Phase 1.

Waterbirds places land/water birds on land/water backgrounds; the background is the spurious
feature and the worst (minority) group is bird-on-opposite-background. The standard
``metadata.csv`` (from the WILDS / group_DRO release) gives (y, place) per image, which yields
group_id = 2*y + place and minority = (y != place).

This file is intentionally a thin stub: Phase 1 is gated on CLEVR-Hans3 + synthetic only. The
loader is wired to the same ImageDatasetBundle interface so Phase 2 needs no re-engineering.
"""
from __future__ import annotations

from typing import Optional

from .base import ImageDatasetBundle, SplitSpec

PHASE = 2


def load_waterbirds(cfg: dict, seed: int, split_spec: Optional[SplitSpec] = None) -> ImageDatasetBundle:
    raise NotImplementedError(
        "Waterbirds is Phase-2 scope. Phase 1 is gated on CLEVR-Hans3 + synthetic. "
        "Implement after the human 'go'. The ImageDatasetBundle interface and group-label "
        "convention (group_id = 2*y + place, minority = y != place) are already defined in "
        "data/base.py so this requires only metadata.csv parsing + a torchvision transform."
    )
