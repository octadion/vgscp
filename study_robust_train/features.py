"""Frozen backbone feature extractors for the two study representations.

  clip_vitb32   : frozen CLIP ViT-B/32 (512-d), reuses the audit-verified cached path.
  resnet50_erm  : train an ERM ResNet-50 on the in-domain composited TRAIN split, then extract
                  the 2048-d penultimate features for every split (the canonical DFR setup that
                  makes worst-group numbers comparable to the group-robustness literature).

The ResNet path is Colab-only (needs torch + GPU + images); it is import-light here and raises a
clear error if torch/images are unavailable. Features are cached to disk and L2-normalized (so the
study's `assert_l2_normalized` §2a guard holds for both backbones).
"""
from __future__ import annotations

import os

import numpy as np

from experiments.real_data import _paths_hash, clip_image_features

__all__ = ["l2_normalize", "clip_features", "resnet50_erm_features"]


def l2_normalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (X / n).astype(np.float32)


def clip_features(paths: list, *, model_name="ViT-B-32", pretrained="openai", device="cuda",
                  cache_dir="results/cache_clip", tag="") -> np.ndarray:
    """L2-normalized frozen CLIP ViT-B/32 features (cached). Reuses experiments.real_data."""
    return clip_image_features(paths, model_name, pretrained, device, cache_dir=cache_dir, tag=tag)


def resnet50_erm_features(paths_by_split: dict, y_by_split: dict, *, train_split="train",
                          cache_dir="results/cache_resnet", tag="", device="cuda",
                          epochs=10, lr=1e-3, batch_size=128, image_size=224, seed=0,
                          max_train=None) -> dict:
    """Train ERM ResNet-50 on ``train_split`` images, return {split: (N,2048) L2-normalized feats}.

    Standard ERM (no group info): ImageNet-pretrained ResNet-50, fc -> n_classes, cross-entropy.
    Penultimate (post-avgpool, pre-fc) features are extracted for ALL splits and cached. Colab-only.

    ``max_train`` (spec §3 budget): if set, the ERM backbone is trained on a seeded RANDOM
    subsample of ``max_train`` train images. Random (NOT class-balanced) is deliberate — it
    PRESERVES the in-domain composited spurious correlation the ERM head must learn (class-
    balancing would distort it). Feature EXTRACTION still covers every split in full; only the
    backbone's training set is subsampled. The §2 worst-group accuracy gate re-verifies sanity.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models, transforms
        from torchvision.models import ResNet50_Weights
        from PIL import Image
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"resnet50_erm_features needs torch/torchvision/PIL (Colab): {e}")

    os.makedirs(cache_dir, exist_ok=True)
    all_paths = [p for sp in paths_by_split for p in paths_by_split[sp]]
    key = f"resnet50erm_{tag}_{_paths_hash(all_paths)}_{epochs}ep_{seed}".replace("/", "-")
    cache_path = os.path.join(cache_dir, f"{key}.npz")
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        return {sp: z[sp] for sp in paths_by_split}

    weights = ResNet50_Weights.IMAGENET1K_V2
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(image_size),
        transforms.ToTensor(), weights.transforms().__class__().normalize
        if hasattr(weights.transforms(), "normalize") else transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    class _DS(Dataset):
        def __init__(self, paths, labels):
            self.paths, self.labels = paths, labels
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, i):
            img = Image.open(self.paths[i]).convert("RGB")
            return tf(img), int(self.labels[i])

    torch.manual_seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    n_classes = int(len(np.unique(y_by_split[train_split])))
    net = models.resnet50(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, n_classes)
    net = net.to(dev)

    # ERM training on the in-domain composited train split (optionally random-subsampled to budget)
    tr_paths = list(paths_by_split[train_split])
    tr_y = np.asarray(y_by_split[train_split])
    if max_train is not None and len(tr_paths) > max_train:
        sub = np.random.default_rng(seed).choice(len(tr_paths), size=int(max_train), replace=False)
        tr_paths = [tr_paths[i] for i in sub]
        tr_y = tr_y[sub]
        print(f"[resnet50-erm {tag}] training on random subsample {len(tr_paths)}/"
              f"{len(paths_by_split[train_split])} (in-domain preserved; §2 gate re-verifies)")
    tr = DataLoader(_DS(tr_paths, tr_y), batch_size=batch_size, shuffle=True, num_workers=2)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    net.train()
    for ep in range(epochs):
        run = 0.0
        for xb, yb in tr:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = crit(net(xb), yb)
            loss.backward(); opt.step()
            run += float(loss)
        print(f"[resnet50-erm {tag}] epoch {ep+1}/{epochs} loss={run/max(1,len(tr)):.4f}")

    # penultimate-feature extractor (drop fc)
    feat_net = nn.Sequential(*list(net.children())[:-1]).to(dev).eval()
    out = {}
    with torch.no_grad():
        for sp, paths in paths_by_split.items():
            dl = DataLoader(_DS(paths, y_by_split[sp]), batch_size=batch_size, shuffle=False,
                            num_workers=2)
            feats = []
            for xb, _ in dl:
                f = feat_net(xb.to(dev)).squeeze(-1).squeeze(-1).cpu().numpy()
                feats.append(f)
            out[sp] = l2_normalize(np.concatenate(feats, axis=0))
    np.savez(cache_path, **out)
    return out
