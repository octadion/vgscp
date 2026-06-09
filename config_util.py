"""Minimal config loader with Hydra-style ``defaults`` merging, without requiring Hydra.

Reads a YAML config; if it has a top-level ``defaults: [name, ...]`` list, loads each
``configs/<name>.yaml`` and shallow-merges their top-level keys underneath the main file
(main file wins on conflicts). Keeps the theory testbed runnable with only PyYAML installed.
"""
from __future__ import annotations

import os
import re
from typing import Any

import yaml

# OmegaConf-style env interpolation ``${oc.env:NAME,default}`` / ``${oc.env:NAME}`` resolved here so
# configs work without Hydra/OmegaConf (the real loaders need WATERBIRDS_ROOT / CUB_ROOT resolved).
_ENV_RE = re.compile(r"\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)(?:,([^}]*))?\}")


def _resolve_env_str(s: str) -> str:
    def sub(m):
        name, default = m.group(1), m.group(2)
        return os.environ.get(name, default if default is not None else "")
    return _ENV_RE.sub(sub, s)


def _resolve_env(obj: Any) -> Any:
    if isinstance(obj, str):
        return _resolve_env_str(obj)
    if isinstance(obj, dict):
        return {k: _resolve_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env(v) for v in obj]
    return obj


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> dict:
    path = os.path.abspath(path)
    cfg_dir = os.path.dirname(path)
    with open(path, "r") as f:
        main = yaml.safe_load(f) or {}
    defaults = main.pop("defaults", []) or []
    merged: dict[str, Any] = {}
    for name in defaults:
        dep_path = os.path.join(cfg_dir, f"{name}.yaml")
        with open(dep_path, "r") as f:
            dep = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, dep)
    merged = _deep_merge(merged, main)
    return _resolve_env(merged)
