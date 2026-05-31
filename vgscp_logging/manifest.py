"""Run manifest + environment / GPU / git capture (Sections 10, 15).

Every run persists: config YAML, env (pip freeze), git hash, GPU model, seeds, split indices,
timings, achieved throughput. This makes every reported number traceable and reproducible.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Optional


def _safe_run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def capture_git() -> dict:
    return {
        "commit": _safe_run(["git", "rev-parse", "HEAD"]),
        "branch": _safe_run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(_safe_run(["git", "status", "--porcelain"])),
    }


def capture_pip_freeze() -> list[str]:
    out = _safe_run([sys.executable, "-m", "pip", "freeze"])
    return out.splitlines() if out else []


def capture_gpu() -> dict:
    info: dict[str, Any] = {"cuda_available": False}
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["capability"] = list(torch.cuda.get_device_capability(0))
            free, total = torch.cuda.mem_get_info()
            info["vram_total_mb"] = total / (1024**2)
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        info["torch_version"] = None
    return info


def capture_env() -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git": capture_git(),
        "gpu": capture_gpu(),
        "pip_freeze": capture_pip_freeze(),
    }


def make_run_id(experiment_name: str, timestamp: Optional[str] = None) -> str:
    """Deterministic-friendly run id. Pass a timestamp string (Date.now is avoided in scripts)."""
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{experiment_name}_{ts}"


class RunManifest:
    """Collects everything needed to reproduce and audit a run; written as manifest.json."""

    def __init__(self, run_dir: str, config: dict, perf_ctx: Optional[dict] = None):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
        self.data: dict[str, Any] = {
            "config": config,
            "perf_context": perf_ctx or {},
            "env": capture_env(),
            "seeds": [],
            "splits": {},
            "throughput": [],
            "stage_paths": {},
            "verdict": None,
        }

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def set_splits(self, split_indices: dict[str, list]) -> None:
        # store sizes inline; full indices saved to a separate npz to keep manifest small
        self.data["splits"] = {k: {"n": len(v)} for k, v in split_indices.items()}
        try:
            import numpy as np

            np.savez(
                os.path.join(self.run_dir, "split_indices.npz"),
                **{k: np.asarray(v) for k, v in split_indices.items()},
            )
            self.data["splits"]["_indices_file"] = "split_indices.npz"
        except ImportError:
            pass

    def set_throughput(self, throughput_list: list) -> None:
        self.data["throughput"] = throughput_list

    def write_config_yaml(self, config: dict) -> None:
        try:
            import yaml

            with open(os.path.join(self.run_dir, "config.yaml"), "w") as f:
                yaml.safe_dump(config, f, sort_keys=False)
        except Exception:
            with open(os.path.join(self.run_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)

    def save(self) -> str:
        # also dump pip freeze to its own file for convenience
        freeze = self.data["env"].get("pip_freeze", [])
        with open(os.path.join(self.run_dir, "pip_freeze.txt"), "w") as f:
            f.write("\n".join(freeze))
        path = os.path.join(self.run_dir, "manifest.json")
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)
        return path
