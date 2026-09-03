"""Colab dataset preparation (download / extract / normalize) for Waterbirds + CelebA.

Importable locally (no torch); the network/credential steps only execute when actually invoked on
Colab. The CelebA NORMALIZATION (locate metadata via os.walk + convert kaggle CSV -> the .txt the
loader expects) is pure and unit-tested in tests/test_colab_data.py. Shared by the grid and
recoverability notebooks so the CelebA prep logic lives in ONE debugged place.
"""
from __future__ import annotations

import os
import subprocess

__all__ = ["prepare_waterbirds", "prepare_celeba", "normalize_celeba", "celeba_ready"]


def _sh(cmd: str):
    print("$", cmd)
    return subprocess.run(cmd, shell=True)


def prepare_waterbirds(drive_cache: str, url: str, dest: str = "/content/data/waterbirds") -> str:
    """wget the Waterbirds tarball (cached on Drive) and extract to ``dest``. Returns ``dest``."""
    os.makedirs(dest, exist_ok=True)
    tb = os.path.join(drive_cache, "waterbirds.tar.gz")
    if not os.path.exists(tb):
        _sh(f"wget -q -O '{tb}' '{url}'")
    _sh(f"tar -xzf '{tb}' -C '{dest}'")
    return dest


def _walk_find(root: str, name: str):
    for dp, _d, fs in os.walk(root):
        if name in fs:
            return os.path.join(dp, name)
    return None


def _walk_any_jpg(root: str):
    for dp, _d, fs in os.walk(root):
        for fn in fs:
            if fn.lower().endswith(".jpg"):
                return os.path.join(dp, fn)
    return None


def celeba_ready(root: str) -> bool:
    return (os.path.exists(f"{root}/list_attr_celeba.txt")
            and os.path.exists(f"{root}/list_eval_partition.txt")
            and _walk_any_jpg(root) is not None)


def normalize_celeba(local: str) -> "str | None":
    """After CelebA images+metadata are extracted under ``local``: locate list_attr / list_eval
    (csv OR txt) and an image via os.walk (layout-agnostic), and convert kaggle CSV metadata to the
    `.txt` format data/celeba.py expects (values -1/1 preserved). Returns ``local`` if ready else
    None. Pure (no network) -> unit-testable."""
    print("[celeba] top-level after unzip:", sorted(os.listdir(local))[:25] if os.path.isdir(local) else "MISSING")
    attr = _walk_find(local, "list_attr_celeba.csv") or _walk_find(local, "list_attr_celeba.txt")
    part = _walk_find(local, "list_eval_partition.csv") or _walk_find(local, "list_eval_partition.txt")
    jpg = _walk_any_jpg(local)
    print(f"[celeba] found attr={attr}\n         part={part}\n         sample_jpg={jpg}")
    if attr is None or part is None or jpg is None:
        print("[celeba] missing a required artifact -> cannot prepare.")
        return None
    import pandas as pd
    if attr.endswith(".csv"):
        df = pd.read_csv(attr); idc = df.columns[0]; names = list(df.columns[1:])
        with open(f"{local}/list_attr_celeba.txt", "w") as f:
            f.write(f"{len(df)}\n{' '.join(names)}\n")
            for _, r in df.iterrows():
                f.write(str(r[idc]) + " " + " ".join(str(int(r[c])) for c in names) + "\n")
    elif attr != f"{local}/list_attr_celeba.txt":
        _sh(f"cp '{attr}' '{local}/list_attr_celeba.txt'")
    if part.endswith(".csv"):
        df = pd.read_csv(part); c0, c1 = df.columns[0], df.columns[1]
        with open(f"{local}/list_eval_partition.txt", "w") as f:
            for _, r in df.iterrows():
                f.write(f"{r[c0]} {int(r[c1])}\n")
    elif part != f"{local}/list_eval_partition.txt":
        _sh(f"cp '{part}' '{local}/list_eval_partition.txt'")
    ok = celeba_ready(local)
    print("[celeba] prepared OK" if ok else "[celeba] STILL incomplete after normalize")
    return local if ok else None


def prepare_celeba(drive_cache: str, *, source: str = "kaggle", celeba_drive: str = "",
                   dest: str = "/content/data/celeba") -> "str | None":
    """Prepare CelebA. source: 'kaggle' (download jessicali9530/celeba-dataset; needs kaggle.json) /
    'drive' (use a pre-extracted folder) / 'skip'. Returns the root path or None."""
    if source == "skip":
        return None
    if source == "drive":
        return celeba_drive if (celeba_drive and os.path.isdir(celeba_drive)) else None
    if source != "kaggle":
        print(f"[celeba] unknown source={source!r}"); return None
    os.makedirs(dest, exist_ok=True)
    if celeba_ready(dest):
        print("[celeba] already prepared at", dest); return dest
    import shutil
    zip_cache = f"{drive_cache}/celeba-dataset.zip"
    big = lambda z: os.path.exists(z) and os.path.getsize(z) > 100 * 1024 * 1024

    # The Drive-cached zip is checked FIRST. kaggle.json is a *download* credential, so demanding
    # it up front made a new session fail even with the 1.4 GB zip already on Drive -- and files
    # under /content do not survive a runtime restart, so that is the common case on a resumed run.
    if big(zip_cache):
        zip_use = zip_cache
        print(f"[celeba] reusing cached zip ({os.path.getsize(zip_cache)/1e9:.2f} GB) "
              f"-- no kaggle.json needed")
        subprocess.run("pip -q install pandas", shell=True)
    else:
        cand = ["kaggle.json", "/content/kaggle.json",
                os.path.expanduser("~/.kaggle/kaggle.json")]
        found = next((c for c in cand if os.path.exists(c)), None)
        if not found:
            print(f"[celeba] no cached zip at {zip_cache} and kaggle.json NOT found. Colab: Files "
                  f"panel -> upload kaggle.json to /content, then re-run. Token: kaggle.com -> "
                  f"Settings -> API -> Create New API Token. (Files in /content are lost on a "
                  f"runtime restart, so re-upload after one.)")
            return None
        os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
        shutil.copy(found, os.path.expanduser("~/.kaggle/kaggle.json"))
        _sh("chmod 600 ~/.kaggle/kaggle.json")
        subprocess.run("pip -q install kaggle pandas", shell=True)
        _sh("kaggle datasets download -d jessicali9530/celeba-dataset -p /content")
        zip_use = "/content/celeba-dataset.zip"
        if big(zip_use):
            _sh(f"cp '{zip_use}' '{zip_cache}'")
        else:
            print(f"[celeba] download FAILED / zip too small (check the token is valid and that "
                  f"you accepted the dataset rules at "
                  f"kaggle.com/datasets/jessicali9530/celeba-dataset). "
                  f"exists={os.path.exists(zip_use)}")
            return None
    _sh(f"unzip -q -o '{zip_use}' -d '{dest}'")
    return normalize_celeba(dest)
