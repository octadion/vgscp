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

__all__ = ["l2_normalize", "clip_features", "resnet50_erm_features", "frozen_features",
           "FROZEN_BACKBONES"]

# Frozen pretrained backbones added for R1.3 ("only two backbones ... limits the
# generalizability"). Chosen to span PRETRAINING REGIMES, not just architectures, so the claim can
# be stated across both axes:
#
#   resnet50_erm    CNN    supervised, fine-tuned in-domain   (existing)
#   clip_vitb32     ViT    image-text contrastive             (existing)
#   dinov2_vitb14   ViT    self-supervised, no labels at all
#   vit_b16_in1k    ViT    supervised ImageNet
#
# vit_b16 is what separates the architecture axis from the pretraining axis: without it "ViT" and
# "not-plain-supervised" are confounded, because both existing ViTs are non-supervised.
#
# Each model uses ITS OWN canonical preprocessing. Forcing one shared transform would quietly
# degrade whichever model did not expect it -- DINOv2 and CLIP do not share ImageNet statistics.
FROZEN_BACKBONES = ("clip_vitb32", "dinov2_vitb14", "vit_b16_in1k")


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
    # Cache check BEFORE the torch import, matching clip_image_features and finetune_features.
    # On a cache hit this function is pure numpy, so a grid re-run over cached features needs
    # neither a GPU nor torch -- which is what lets the analysis stage run on a CPU runtime.
    os.makedirs(cache_dir, exist_ok=True)
    all_paths = [p for sp in paths_by_split for p in paths_by_split[sp]]
    key = f"resnet50erm_{tag}_{_paths_hash(all_paths)}_{epochs}ep_{seed}".replace("/", "-")
    cache_path = os.path.join(cache_dir, f"{key}.npz")
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        return {sp: z[sp] for sp in paths_by_split}

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models, transforms
        from torchvision.models import ResNet50_Weights
        from PIL import Image
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"resnet50_erm_features needs torch/torchvision/PIL (Colab): {e}")

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


def _build_frozen(name: str, device):
    """(model, transform, dim) for one frozen backbone, using its own canonical preprocessing."""
    # Validate before importing torch, so a typo fails immediately and identically everywhere --
    # including on a machine with no torch, where the import would otherwise mask the real error.
    if name not in FROZEN_BACKBONES or name == "clip_vitb32":
        raise ValueError(f"unknown frozen backbone {name!r}; "
                         f"choose from {[b for b in FROZEN_BACKBONES if b != 'clip_vitb32']} "
                         f"(clip_vitb32 has its own loader)")
    import torch
    from torchvision import transforms

    if name == "vit_b16_in1k":
        from torchvision.models import ViT_B_16_Weights, vit_b_16
        w = ViT_B_16_Weights.IMAGENET1K_V1
        net = vit_b_16(weights=w)
        net.heads = torch.nn.Identity()            # keep the 768-d pre-logit representation
        return net.to(device).eval(), w.transforms(), 768

    if name == "dinov2_vitb14":
        # torch.hub, so no new package. DINOv2 is self-supervised: it never saw a label, which is
        # what makes it the most distinct regime in the set.
        net = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
        tf = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        return net.to(device).eval(), tf, 768

    raise AssertionError(f"unreachable: {name!r} passed validation but has no builder")


def frozen_features(paths: list, *, name: str, device: str = "cuda",
                    cache_dir: str = "results/cache_frozen", tag: str = "",
                    batch_size: int = 128, num_workers: int = 8,
                    inflight_bytes: int = 1_500_000_000) -> np.ndarray:
    """L2-normalised frozen features for ``paths`` from backbone ``name`` (cached to disk).

    Cache is checked BEFORE importing torch, so a cached run costs no GPU and no torch -- which is
    what lets the whole grid re-run on a CPU runtime. Worker count is derived from an explicit
    host-RAM budget, because a DataLoader reserves batch x workers x prefetch decoded images.
    """
    if name not in FROZEN_BACKBONES or name == "clip_vitb32":
        raise ValueError(f"unknown frozen backbone {name!r}")
    os.makedirs(cache_dir, exist_ok=True)
    key = f"{name}_{tag}_{_paths_hash(list(paths))}".replace("/", "-")
    cache_path = os.path.join(cache_dir, f"frozen_{key}.npy")
    if os.path.exists(cache_path):
        feats = np.load(cache_path)
        if feats.shape[0] == len(paths):
            return feats
        print(f"[{name} {tag}] cached {feats.shape[0]} rows but {len(paths)} paths; recomputing")

    import torch
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    net, tf, dim = _build_frozen(name, dev)

    class _DS(Dataset):
        def __init__(self, p):
            self.p = list(p)

        def __len__(self):
            return len(self.p)

        def __getitem__(self, i):
            return tf(Image.open(self.p[i]).convert("RGB"))

    img_bytes = 3 * 224 * 224 * 4
    nw = max(2, min(num_workers, int(inflight_bytes / (batch_size * 2 * img_bytes))))
    dl = DataLoader(_DS(paths), batch_size=batch_size, shuffle=False, num_workers=nw,
                    pin_memory=True)
    print(f"[{name} {tag}] extracting {len(paths)} images, bs={batch_size}, workers={nw}",
          flush=True)

    out, i = None, 0
    with torch.no_grad():
        for xb in dl:
            f = net(xb.to(dev, non_blocking=True)).float().cpu().numpy()
            if out is None:                       # preallocate: a list + concatenate would hold
                out = np.empty((len(paths), f.shape[1]), dtype=np.float32)   # two full copies
            out[i:i + f.shape[0]] = f
            i += f.shape[0]
    assert out is not None and i == len(paths), f"extracted {i} of {len(paths)}"
    out = l2_normalize(out)
    tmp = cache_path + ".tmp.npy"
    np.save(tmp, out)
    os.replace(tmp, cache_path)                   # atomic: a truncated file must not pass as cache
    print(f"[{name} {tag}] cached {out.shape} -> {cache_path}", flush=True)
    return out
