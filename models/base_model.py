"""Base classifier f (ERM) — Section 6.

Standard backbone (ResNet-34 for CLEVR-Hans, ResNet-50 for Waterbirds/CelebA) trained with ERM
so the shortcut is present. Exposes logits, softmax probs, and the penultimate feature phi(x)
used by the trust score. torch/torchvision are imported lazily.

Worst-group accuracy is logged after training to CONFIRM a shortcut exists (Section 6): if
worst-group acc is not substantially below overall acc, warn — the regime is wrong.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def build_backbone(name: str, num_classes: int, pretrained: bool = True, dropout_p: float = 0.0):
    """Return (model, feature_dim). Adds an optional dropout before the classifier head so the
    SAME architecture supports MC-dropout without retraining."""
    import torch.nn as nn
    import torchvision

    name = name.lower()
    if name == "resnet34":
        net = torchvision.models.resnet34(
            weights=torchvision.models.ResNet34_Weights.DEFAULT if pretrained else None
        )
        feat_dim = net.fc.in_features
    elif name == "resnet50":
        net = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        feat_dim = net.fc.in_features
    else:
        raise ValueError(f"unknown backbone {name!r}")

    net.fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(feat_dim, num_classes))
    return net, feat_dim


class FeatureClassifier:
    """Thin wrapper exposing forward, features, and probs over a torch backbone."""

    def __init__(self, net, feat_dim: int, device: str = "cuda"):
        import torch  # noqa

        self.net = net.to(device)
        self.feat_dim = feat_dim
        self.device = device

    def _penultimate(self, x):
        """Run the backbone up to (but not including) the fc head -> phi(x)."""
        import torch

        m = self.net
        # ResNet feature extraction up to global avg pool
        z = m.conv1(x)
        z = m.bn1(z)
        z = m.relu(z)
        z = m.maxpool(z)
        z = m.layer1(z)
        z = m.layer2(z)
        z = m.layer3(z)
        z = m.layer4(z)
        z = m.avgpool(z)
        return torch.flatten(z, 1)

    def logits_and_features(self, x):
        feats = self._penultimate(x)
        logits = self.net.fc(feats)
        return logits, feats


@dataclass
class TrainConfig:
    num_epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"


def train_erm(net, loader, train_cfg: TrainConfig, perf_ctx, device: str = "cuda"):
    """Standard ERM training loop. Kept minimal; AMP via perf_ctx for the forward/backward."""
    import torch
    import torch.nn as nn
    from tqdm import tqdm

    net.train().to(device)
    if train_cfg.optimizer == "adam":
        opt = torch.optim.Adam(net.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    else:
        opt = torch.optim.SGD(net.parameters(), lr=train_cfg.lr, momentum=0.9,
                              weight_decay=train_cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(perf_ctx.precision == "fp16"))
    crit = nn.CrossEntropyLoss()
    for epoch in range(train_cfg.num_epochs):
        for x, y in tqdm(loader, desc=f"ERM epoch {epoch+1}/{train_cfg.num_epochs}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if perf_ctx.channels_last:
                x = x.to(memory_format=torch.channels_last)
            opt.zero_grad(set_to_none=True)
            amp_dtype = perf_ctx.amp_dtype
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=perf_ctx.use_amp):
                out = net(x)
                loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
    return net


def worst_group_accuracy(y_pred: np.ndarray, y_true: np.ndarray, group_id: np.ndarray) -> dict:
    """Per-group + worst-group accuracy. Used to confirm a shortcut exists (Section 6)."""
    accs = {}
    for g in np.unique(group_id):
        m = group_id == g
        accs[int(g)] = float((y_pred[m] == y_true[m]).mean())
    overall = float((y_pred == y_true).mean())
    worst = min(accs.values()) if accs else float("nan")
    return {"per_group": accs, "overall": overall, "worst_group": worst}
