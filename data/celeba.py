"""CelebA (Blond x Gender) loader — PHASE 2 OPTIONAL. Stub for Phase 1.

Spurious attribute = gender (Male), target = hair color (Blond). The minority group is
blond males / non-blond females depending on the labeling convention; group_id = 2*y + gender,
minority = (y != gender) under the standard subpopulation-shift setup.

Intentionally a stub — Phase 1 is gated on CLEVR-Hans3 + synthetic. Wired to the same
ImageDatasetBundle interface so Phase 2 needs no re-engineering.
"""
from __future__ import annotations

from typing import Optional

from .base import ImageDatasetBundle, SplitSpec

PHASE = 2


def load_celeba(cfg: dict, seed: int, split_spec: Optional[SplitSpec] = None) -> ImageDatasetBundle:
    raise NotImplementedError(
        "CelebA is Phase-2 (optional) scope. Implement after the human 'go'."
    )
