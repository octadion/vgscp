"""Unit tests for the pure CelebA normalization core (no network/torch)."""
import os

import pandas as pd

from study_robust_train.colab_data import celeba_ready, normalize_celeba
from data.celeba import load_celeba


def _kaggle_layout(root):
    # kaggle: images nested two levels, CSV metadata at root
    imgdir = os.path.join(root, "img_align_celeba", "img_align_celeba"); os.makedirs(imgdir)
    fns = [f"{i:06d}.jpg" for i in range(1, 9)]
    for fn in fns:
        open(os.path.join(imgdir, fn), "wb").close()
    attrs = ["Blond_Hair", "Male", "Smiling"]
    vals = [[1, -1, 1], [1, 1, -1], [-1, 1, 1], [-1, -1, -1], [1, -1, 1], [-1, 1, -1], [1, 1, 1], [-1, -1, 1]]
    pd.DataFrame([[fn] + v for fn, v in zip(fns, vals)], columns=["image_id"] + attrs).to_csv(
        f"{root}/list_attr_celeba.csv", index=False)
    pd.DataFrame({"image_id": fns, "partition": [0, 0, 0, 0, 1, 1, 2, 2]}).to_csv(
        f"{root}/list_eval_partition.csv", index=False)
    return fns


def test_normalize_kaggle_csv_then_loader(tmp_path):
    root = str(tmp_path)
    _kaggle_layout(root)
    assert not celeba_ready(root)                 # csv only, no .txt yet
    out = normalize_celeba(root)
    assert out == root and celeba_ready(root)
    assert os.path.exists(f"{root}/list_attr_celeba.txt")
    assert os.path.exists(f"{root}/list_eval_partition.txt")
    # the real loader must parse the converted txt + locate nested images
    b = load_celeba({"root": root, "n_classes": 2}, seed=0, build_datasets=False)
    import numpy as np
    g = np.concatenate([b.group_id[s] for s in ("train", "d_learn", "d_cal", "d_test")])
    y = np.concatenate([b.y[s] for s in ("train", "d_learn", "d_cal", "d_test")])
    m = np.concatenate([b.spurious_attr[s] for s in ("train", "d_learn", "d_cal", "d_test")])
    assert np.array_equal(g, 2 * y + m)
    assert 3 in set(int(x) for x in g)            # rare blond-male group present
    assert os.path.exists(b.meta["paths"]["train"][0])


def test_normalize_missing_artifact_returns_none(tmp_path):
    # only metadata, no images -> not ready
    root = str(tmp_path)
    open(f"{root}/list_attr_celeba.txt", "w").close()
    open(f"{root}/list_eval_partition.txt", "w").close()
    assert normalize_celeba(root) is None


def test_cached_zip_does_not_require_kaggle_json(tmp_path, monkeypatch):
    """A Drive-cached zip must be usable without a credential.

    kaggle.json is a *download* credential. Demanding it before checking the cache made a resumed
    session fail with the 1.4 GB zip already on Drive -- and files under /content are lost on a
    runtime restart, so that is the normal state of a second session.
    """
    import subprocess as sp
    import types
    from study_robust_train import colab_data as cd

    drive = tmp_path / "drive"
    drive.mkdir()
    (drive / "celeba-dataset.zip").write_bytes(b"x" * (101 * 1024 * 1024))

    calls = []
    monkeypatch.setattr(cd, "_sh", lambda c: calls.append(c))
    monkeypatch.setattr(cd, "normalize_celeba", lambda d: d)
    monkeypatch.setattr(sp, "run",
                        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="", stderr=""))
    # no kaggle.json anywhere: point HOME at an empty dir and run from one too
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)

    out = cd.prepare_celeba(str(drive), source="kaggle", dest=str(tmp_path / "out"))
    assert out is not None, "cached zip should be used without a credential"
    assert any("unzip" in c for c in calls), "should have unzipped the cached zip"
    assert not any("kaggle datasets download" in c for c in calls), "must not re-download"


def test_no_zip_and_no_credential_fails_clearly(tmp_path, monkeypatch):
    """With neither a cached zip nor a credential, it must decline rather than half-proceed."""
    from study_robust_train import colab_data as cd

    monkeypatch.setattr(cd, "_sh", lambda c: None)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    assert cd.prepare_celeba(str(tmp_path / "empty_drive"), source="kaggle",
                             dest=str(tmp_path / "out")) is None
