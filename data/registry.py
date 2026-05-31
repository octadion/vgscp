"""Dataset registry — name -> loader. Synthetic is handled by theory/ directly."""
from __future__ import annotations

from typing import Optional

from .base import SplitSpec


def load_dataset(name: str, cfg: dict, seed: int, split_spec: Optional[SplitSpec] = None):
    if name == "clevr_hans3":
        from .clevr_hans import load_clevr_hans3

        return load_clevr_hans3(cfg, seed, split_spec)
    if name == "waterbirds":
        from .waterbirds import load_waterbirds

        return load_waterbirds(cfg, seed, split_spec)
    if name == "celeba":
        from .celeba import load_celeba

        return load_celeba(cfg, seed, split_spec)
    raise ValueError(f"unknown dataset {name!r}")
